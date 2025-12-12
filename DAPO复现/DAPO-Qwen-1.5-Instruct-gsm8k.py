
# 基于Qwen2.5-1.5B-Instruct模型，使用DAPO算法在gsm8k数据集上进行数学问题求解的强化学习微调
# 使用到A100 GPU进行训练

import random
import copy
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def set_random_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(42)

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_answer_from_model_output(text: str):
    """提取模型输出中的答案"""
    parts = text.split("<answer>")
    if len(parts) < 2:
        return None
    last_part = parts[-1]
    if "</answer>" not in last_part:
        return None
    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer


def extract_answer_from_dataset(text: str):
    """提取数据集中的答案"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def build_prompt(messages: List[Dict[str, str]]):
    """构建提示"""
    return "\n".join([msg["content"].strip() for msg in messages])


def prepare_dataset(split="train"):
    """准备数据集"""
    data = load_dataset("openai/gsm8k", "main")[split]
    formatted_data = []
    for example in data:
        prompt_str = build_prompt(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ]
        )
        formatted_example = {
            "prompt": prompt_str,
            "answer": extract_answer_from_dataset(example["answer"]),
        }
        formatted_data.append(formatted_example)
    return formatted_data


def extract_last_number(text: str):
    """提取文本中的最后一个数字"""
    text = text.replace("$", "").replace("%", "")
    pattern = r"(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def extract_single_number(text: str):
    """提取文本中的单个数字"""
    numbers = re.findall(r"-?\d*\.?\d+", text)
    return float(numbers[0]) if len(numbers) == 1 else None


def evaluate_model(model, tokenizer, eval_examples, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "=" * 50)
    print("EVALUATION ON", total, "EXAMPLES")
    print("=" * 50)

    for example in eval_examples:
        full_prompt = example["prompt"]
        expected = example["answer"]

        inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=300,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,
                early_stopping=False,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            predicted = extract_answer_from_model_output(response)

            if predicted == expected:
                is_correct = True
            else:
                pred_num = extract_single_number(str(predicted))
                exp_num = extract_single_number(str(expected))
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    pred_num = extract_last_number(str(predicted))
                    exp_num = extract_last_number(str(expected))
                    is_correct = (
                        pred_num is not None
                        and exp_num is not None
                        and pred_num == exp_num
                    )

            if is_correct:
                correct += 1

            print("\nPrompt:")
            print(full_prompt)
            print("\nExpected Answer:")
            print(expected)
            print("\nExtracted Answer:")
            print(predicted)
            print("\nFull Generated Response:")
            print(response)
            print("\nCorrect:", "✓" if is_correct else "✗")
            print("-" * 50)

        except Exception as e:
            print("\nFailed to parse model output for prompt:")
            print(full_prompt)
            print("Error:", e)
            print("-" * 50)

    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    print("=" * 50)
    model.train()
    return accuracy


def correctness_reward(prompts, completions, answer, **kwargs):
    """计算正确性奖励"""
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_answer_from_model_output(r) for r in responses]
    rewards = []
    for r, a in zip(extracted, answer):
        if r == a:
            rewards.append(2.0)
        else:
            r_num = extract_single_number(str(r))
            a_num = extract_single_number(str(a))
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards.append(1.5)
            else:
                rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs):
    """计算格式奖励"""
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response:
            score += 0.2
        if "</reasoning>" in response:
            score += 0.2
        if "<answer>" in response:
            score += 0.2
        if "</answer>" in response:
            score += 0.2
        rewards.append(score)
    return rewards


def combined_reward(prompts, completions, answer):
    """计算综合奖励"""
    correctness_scores = correctness_reward(
        prompts=prompts, completions=completions, answer=answer
    )
    format_scores = format_reward(completions=completions)
    return [c + f for c, f in zip(correctness_scores, format_scores)]


def selective_log_softmax(logits, input_ids):
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, input_ids)


def create_completion_mask(completion_ids, eos_token_id):
    """创建完成掩码：到首次 eos 为止（含 eos）为 1，其余为 0"""
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full(
        (is_eos.size(0),),
        is_eos.size(1),
        dtype=torch.long,
        device=completion_ids.device,
    )
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = (
        torch.arange(is_eos.size(1), device=completion_ids.device)
        .expand(is_eos.size(0), -1)
    )
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()


# -----------------------------
# 1) Soft punishment (DAPO)
# -----------------------------
def soft_overlong_punishment(lengths: torch.Tensor, L_max: int, L_cache: int) -> torch.Tensor:
    """
    软长度惩罚
    """
    if L_cache <= 0:
        return torch.zeros_like(lengths, dtype=torch.float32)

    lengths_f = lengths.to(torch.float32)
    L_max_f = float(L_max)
    L_cache_f = float(L_cache)
    L_expected_f = float(L_max - L_cache)

    penalty = torch.zeros_like(lengths_f)
    in_soft = (lengths_f > L_expected_f) & (lengths_f <= L_max_f)
    penalty[in_soft] = (L_expected_f - lengths_f[in_soft]) / L_cache_f
    penalty[lengths_f > L_max_f] = -1.0
    return torch.clamp(penalty, min=-1.0, max=0.0)


def apply_overlong_reward_shaping(
    correctness_rewards, completion_mask, expected_max_len, cache_len
):
    """
    将 soft punishment 加到 correctness reward 上（只惩罚长度，不影响 format reward）
    expected_max_len: 论文中的 L_expected
    cache_len: L_cache
    生成上限 L_max = expected + cache
    """
    lengths = completion_mask.sum(dim=1)  # [B*G]
    L_max = int(expected_max_len + cache_len)
    penalty = soft_overlong_punishment(lengths, L_max=L_max, L_cache=int(cache_len))

    if isinstance(correctness_rewards, torch.Tensor):
        corr = correctness_rewards.to(torch.float32)
    else:
        corr = torch.tensor(correctness_rewards, dtype=torch.float32, device=penalty.device)

    shaped_correctness = corr + penalty.to(corr.device)
    return shaped_correctness, penalty.to(corr.device)


def generate_completions(
    model, tokenizer, prompts, num_generations=8, max_completion_length=256
):
    """
    仅用于 rollout / 采样：强制 eval + no_grad
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    prompt_length = prompt_ids.size(1)

    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

    with torch.no_grad():
        outputs = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )

    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    return prompt_ids, prompt_mask, completion_ids, completion_mask


def dapo_loss(
    model, rollout_data, beta=0.01, epsilon_low=0.2, epsilon_high=0.28
):
    """DAPO损失：Clip-Higher + Token-Level Loss（你原实现的形式）"""
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]

    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    ratio = torch.exp(token_log_probs - old_log_probs)

    rewards = rollout_data["rewards"]
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]

    rewards = rewards.view(batch_size, num_generations)
    avg_reward = rewards.mean().item()

    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)
    advantages = advantages.view(-1, 1)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages
    surrogate_loss = torch.min(surr1, surr2)

    valid_token_sum = completion_mask.sum()
    if valid_token_sum == 0:
        device = input_ids.device
        return torch.tensor(0.0, requires_grad=True, device=device), avg_reward

    loss = -(surrogate_loss * completion_mask).sum() / valid_token_sum
    return loss, avg_reward


def optimize_model_memory(model):
    """优化模型内存（梯度检查点 + 输入 embedding requires_grad）"""
    model.train()
    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.gradient_checkpointing_enable()
    return model


# -----------------------------
# 2) Dynamic Sampling: fill effective batch
#    valid = reward variance > 0 (not all-same reward)
# -----------------------------
def train_with_dapo(
    model,
    tokenizer,
    train_data,
    num_iterations=1,
    num_steps=500,
    batch_size=4,
    num_generations=8,
    max_completion_length=256,   # L_expected
    learning_rate=5e-6,
    mu=3,
    epsilon_low=0.2,
    epsilon_high=0.28,
    device_ids=None,
    overlong_cache_len=None,     # L_cache
    dynamic_sampling_candidate_batch_size=None,
    max_sampling_rounds=200,     # 防止采样卡死
):
    assert device_ids is not None, "Need GPUs (set device_ids)."

    device = torch.device("cuda:0")
    if overlong_cache_len is None:
        overlong_cache_len = max(1, max_completion_length // 5)
    if dynamic_sampling_candidate_batch_size is None:
        dynamic_sampling_candidate_batch_size = max(batch_size, batch_size * 2)

    gen_max_new_tokens = int(max_completion_length + overlong_cache_len)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # DataParallel
    model = nn.DataParallel(model, device_ids=device_ids)
    print(f"Model wrapped with DataParallel across GPUs: {device_ids}")
    print(
        f"Overlong shaping: expected_max={max_completion_length}, "
        f"cache={overlong_cache_len}, gen_max={gen_max_new_tokens}"
    )

    # 关键：DP 之后再启用 memory 优化，确保 replica 也生效
    model.module = optimize_model_memory(model.module)

    metrics = {
        "loss": [],
        "avg_reward": [],
        "avg_correctness": [],
        "avg_format": [],
        "avg_len_penalty": [],
    }

    def flatten_promptwise(list_2d):
        return [x for row in list_2d for x in row]

    for iteration in range(num_iterations):
        # ref_model 保留但不用于 KL（你原设定）
        ref_model = copy.deepcopy(model.module)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        step = 0
        while step < num_steps:
            buffer_items = []
            rounds = 0

            # fill effective batch
            while len(buffer_items) < batch_size:
                rounds += 1
                if rounds > max_sampling_rounds:
                    # 兜底：避免死循环。此时直接放宽条件，接受当前采到的任意 prompt
                    # （不会影响可运行性；同时打印告警，方便你排查 reward/采样设置）
                    print(
                        f"[Warn] Dynamic sampling reached max rounds={max_sampling_rounds}. "
                        f"Collected {len(buffer_items)}/{batch_size}. "
                        f"Falling back to accept any prompts to proceed."
                    )
                    # 补齐：随机补 prompt（即使组内方差为0也接受）
                    while len(buffer_items) < batch_size:
                        s = random.choice(train_data)
                        buffer_items.append({
                            "prompt": s["prompt"],
                            "answer": s["answer"],
                            "completion_ids": None,
                            "completion_mask": None,
                            "formatted_completions": None,
                            "correctness_rewards": None,
                            "format_rewards": None,
                            "penalties": None,
                            "shaped_rewards": None,
                        })
                    break

                candidate_samples = random.sample(
                    train_data, dynamic_sampling_candidate_batch_size
                )
                prompts = [s["prompt"] for s in candidate_samples]
                answers = [s["answer"] for s in candidate_samples]

                # rollout: 使用单卡 generate（更稳定）
                rollout_model = model.module
                _, _, completion_ids, completion_mask = generate_completions(
                    rollout_model,
                    tokenizer,
                    prompts,
                    num_generations=num_generations,
                    max_completion_length=gen_max_new_tokens,
                )

                # decode completions
                formatted_completions = [
                    [{"content": tokenizer.decode(ids, skip_special_tokens=True)}]
                    for ids in completion_ids
                ]
                repeated_prompts = [p for p in prompts for _ in range(num_generations)]
                repeated_answers = [a for a in answers for _ in range(num_generations)]

                # 3) reward decomposition
                corr_rewards = correctness_reward(
                    prompts=repeated_prompts,
                    completions=formatted_completions,
                    answer=repeated_answers,
                )
                fmt_rewards = format_reward(completions=formatted_completions)

                # 1) soft punishment on correctness
                shaped_corr, penalties = apply_overlong_reward_shaping(
                    correctness_rewards=corr_rewards,
                    completion_mask=completion_mask,
                    expected_max_len=max_completion_length,
                    cache_len=overlong_cache_len,
                )
                fmt_t = torch.tensor(
                    fmt_rewards, dtype=torch.float32, device=shaped_corr.device
                )
                shaped_total = shaped_corr + fmt_t  # [B*G]

                # 2) valid = per-prompt reward variance > 0 (not all-same reward)
                B = len(prompts)
                G = num_generations
                shaped_view = shaped_total.view(B, G)
                valid_mask = shaped_view.std(dim=1) > 1e-6
                valid_indices = valid_mask.nonzero(as_tuple=False).view(-1).tolist()
                if len(valid_indices) == 0:
                    continue

                # 将 per-prompt 的 G 条 completion 切出来放入 buffer
                Lcand = completion_ids.size(1)
                comp_ids_3d = completion_ids.view(B, G, Lcand).detach().cpu()
                comp_mask_3d = completion_mask.view(B, G, Lcand).detach().cpu()

                corr_2d = torch.tensor(corr_rewards, dtype=torch.float32).view(B, G).cpu()
                fmt_2d = torch.tensor(fmt_rewards, dtype=torch.float32).view(B, G).cpu()
                pen_2d = penalties.view(B, G).cpu()
                shaped_2d = shaped_total.view(B, G).cpu()

                fmt_comp_2d = [
                    formatted_completions[i * G : (i + 1) * G] for i in range(B)
                ]

                for i in valid_indices:
                    if len(buffer_items) >= batch_size:
                        break
                    buffer_items.append(
                        {
                            "prompt": prompts[i],
                            "answer": answers[i],
                            "completion_ids": comp_ids_3d[i],        # [G, L_i]
                            "completion_mask": comp_mask_3d[i],      # [G, L_i]
                            "formatted_completions": fmt_comp_2d[i], # list len G
                            "correctness_rewards": corr_2d[i].tolist(),
                            "format_rewards": fmt_2d[i].tolist(),
                            "penalties": pen_2d[i].tolist(),
                            "shaped_rewards": shaped_2d[i].tolist(),
                        }
                    )

            # 如果触发兜底补齐，但 completion_ids 为空：重新生成一次（保证可训练）
            if any(it["completion_ids"] is None for it in buffer_items):
                prompts = [it["prompt"] for it in buffer_items]
                answers = [it["answer"] for it in buffer_items]
                rollout_model = model.module
                _, _, completion_ids, completion_mask = generate_completions(
                    rollout_model,
                    tokenizer,
                    prompts,
                    num_generations=num_generations,
                    max_completion_length=gen_max_new_tokens,
                )
                formatted_completions = [
                    [{"content": tokenizer.decode(ids, skip_special_tokens=True)}]
                    for ids in completion_ids
                ]
                repeated_prompts = [p for p in prompts for _ in range(num_generations)]
                repeated_answers = [a for a in answers for _ in range(num_generations)]
                corr_rewards = correctness_reward(
                    prompts=repeated_prompts,
                    completions=formatted_completions,
                    answer=repeated_answers,
                )
                fmt_rewards = format_reward(completions=formatted_completions)
                shaped_corr, penalties = apply_overlong_reward_shaping(
                    correctness_rewards=corr_rewards,
                    completion_mask=completion_mask,
                    expected_max_len=max_completion_length,
                    cache_len=overlong_cache_len,
                )
                fmt_t = torch.tensor(fmt_rewards, dtype=torch.float32, device=shaped_corr.device)
                shaped_total = shaped_corr + fmt_t

                B = batch_size
                G = num_generations
                Lcand = completion_ids.size(1)

                comp_ids_3d = completion_ids.view(B, G, Lcand).detach().cpu()
                comp_mask_3d = completion_mask.view(B, G, Lcand).detach().cpu()
                corr_2d = torch.tensor(corr_rewards, dtype=torch.float32).view(B, G).cpu()
                fmt_2d = torch.tensor(fmt_rewards, dtype=torch.float32).view(B, G).cpu()
                pen_2d = penalties.view(B, G).cpu()
                shaped_2d = shaped_total.view(B, G).cpu()

                fmt_comp_2d = [
                    formatted_completions[i * G : (i + 1) * G] for i in range(B)
                ]
                for i in range(B):
                    buffer_items[i]["completion_ids"] = comp_ids_3d[i]
                    buffer_items[i]["completion_mask"] = comp_mask_3d[i]
                    buffer_items[i]["formatted_completions"] = fmt_comp_2d[i]
                    buffer_items[i]["correctness_rewards"] = corr_2d[i].tolist()
                    buffer_items[i]["format_rewards"] = fmt_2d[i].tolist()
                    buffer_items[i]["penalties"] = pen_2d[i].tolist()
                    buffer_items[i]["shaped_rewards"] = shaped_2d[i].tolist()

            # -------------------------
            # Build one training batch
            # -------------------------
            batch_prompts = [it["prompt"] for it in buffer_items]
            batch_answers = [it["answer"] for it in buffer_items]

            # prompt tokenize
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, padding_side="left")
            prompt_ids = inputs["input_ids"].to(device)
            prompt_mask = inputs["attention_mask"].to(device)
            prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
            prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

            # pad completions to global max L
            pad_id = tokenizer.pad_token_id
            max_L = max(it["completion_ids"].shape[1] for it in buffer_items)

            comp_ids_list = []
            comp_mask_list = []
            formatted_completions_promptwise = []
            rewards_promptwise = []
            corr_promptwise = []
            fmt_promptwise = []
            pen_promptwise = []

            for it in buffer_items:
                ids = it["completion_ids"]          # [G, L_i]
                msk = it["completion_mask"]         # [G, L_i]
                Li = ids.shape[1]
                if Li < max_L:
                    pad_len = max_L - Li
                    ids = torch.cat([ids, torch.full((num_generations, pad_len), pad_id, dtype=ids.dtype)], dim=1)
                    msk = torch.cat([msk, torch.zeros((num_generations, pad_len), dtype=msk.dtype)], dim=1)
                comp_ids_list.append(ids)
                comp_mask_list.append(msk)

                formatted_completions_promptwise.append(it["formatted_completions"])
                rewards_promptwise.append(it["shaped_rewards"])
                corr_promptwise.append(it["correctness_rewards"])
                fmt_promptwise.append(it["format_rewards"])
                pen_promptwise.append(it["penalties"])

            completion_ids = torch.cat(comp_ids_list, dim=0).to(device)      # [B*G, L]
            completion_mask = torch.cat(comp_mask_list, dim=0).to(device)    # [B*G, L]

            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

            # old_log_probs: eval + no_grad 避免 checkpoint 警告
            model.eval()
            with torch.no_grad():
                old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)

            # flatten lists for logging / reward functions
            formatted_completions = flatten_promptwise(formatted_completions_promptwise)
            repeated_prompts = [p for p in batch_prompts for _ in range(num_generations)]
            repeated_answers = [a for a in batch_answers for _ in range(num_generations)]

            rewards_flat = torch.tensor(flatten_promptwise(rewards_promptwise), dtype=torch.float32, device=device)

            rollout_data = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "completion_mask": completion_mask,
                "old_log_probs": old_log_probs,
                "formatted_completions": formatted_completions,
                "repeated_prompts": repeated_prompts,
                "repeated_answers": repeated_answers,
                "logits_to_keep": logits_to_keep,
                "batch_size": batch_size,
                "num_generations": num_generations,
                "rewards": rewards_flat,
            }

            # metrics (3) decomposition
            avg_corr = float(np.mean(flatten_promptwise(corr_promptwise)))
            avg_fmt = float(np.mean(flatten_promptwise(fmt_promptwise)))
            avg_pen = float(np.mean(flatten_promptwise(pen_promptwise)))
            avg_total = float(np.mean(flatten_promptwise(rewards_promptwise)))

            metrics["avg_reward"].append(avg_total)
            metrics["avg_correctness"].append(avg_corr)
            metrics["avg_format"].append(avg_fmt)
            metrics["avg_len_penalty"].append(avg_pen)

            step += 1

            # GRPO-style multiple updates
            for grpo_iter in range(mu):
                model.train()
                loss, avg_reward_view = dapo_loss(
                    model,
                    rollout_data,
                    beta=0.0,  # 你原本移除了 KL；这里保留接口但不使用
                    epsilon_low=epsilon_low,
                    epsilon_high=epsilon_high,
                )
                metrics["loss"].append(float(loss.item()))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if grpo_iter == mu - 1:
                    print(
                        f"Step {step}, Loss: {loss.item():.4f}, "
                        f"AvgReward(total_shaped): {avg_total:.4f}, "
                        f"AvgReward(model_view): {avg_reward_view:.4f}, "
                        f"AvgCorrectness: {avg_corr:.4f}, AvgFormat: {avg_fmt:.4f}, "
                        f"AvgLenPenalty: {avg_pen:.4f}, EffectiveBatch: {batch_size}"
                    )

    # 保存指标
    try:
        np.savez(
            "dapo_training_metrics.npz",
            loss=np.array(metrics["loss"], dtype=np.float32),
            avg_reward=np.array(metrics["avg_reward"], dtype=np.float32),
            correctness_reward=np.array(metrics["avg_correctness"], dtype=np.float32),
            format_reward=np.array(metrics["avg_format"], dtype=np.float32),
            avg_len_penalty=np.array(metrics["avg_len_penalty"], dtype=np.float32),
        )
        print("Saved metrics to dapo_training_metrics.npz")
    except Exception as e:
        print(f"Warning: failed to save metrics: {e}")

    return model


# -----------------------------
# Main
# -----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using primary device: {device}")

model_path = "/root/ysm/shiyan/models/Qwen2.5-1.5B-Instruct"
print("loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
)
model.to(device)
print("Model loaded")

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPUs")
device_ids = list(range(num_gpus)) if num_gpus > 1 else [0]

all_data = prepare_dataset("train")
random.shuffle(all_data)
size_of_eval_data = 30
eval_data = all_data[:size_of_eval_data]
train_data = all_data[size_of_eval_data:]

print("\nInitial model evaluation before finetuning:")
pre_acc = evaluate_model(model, tokenizer, eval_data, device)
print(f"Pre-DAPO Accuracy: {pre_acc:.2f}%")

# 先对原模型做内存优化（DP 内还会再做一次，确保 replica 生效）
model = optimize_model_memory(model)

print("\nStarting RL fine-tuning using DAPO...")

training_config = {
    "num_iterations": 1,
    "num_steps": 500,
    "batch_size": 4,
    "num_generations": 8,
    "max_completion_length": 256,   # L_expected
    "learning_rate": 5e-6,
    "mu": 3,
    "epsilon_high": 0.28,
    "epsilon_low": 0.2,
    # 可选调参：
    "overlong_cache_len": 51,
    "dynamic_sampling_candidate_batch_size": 16,
    "max_sampling_rounds": 200,
}

model = train_with_dapo(
    model=model,
    tokenizer=tokenizer,
    train_data=train_data,
    device_ids=device_ids,
    **training_config,
)

# 解包 DataParallel
if isinstance(model, torch.nn.DataParallel):
    print("Unwrapping model from DataParallel...")
    model = model.module

print("\nFinal model evaluation after DAPO RL fine-tuning:")
post_acc = evaluate_model(model, tokenizer, eval_data, device)
print(f"Post-DAPO Accuracy: {post_acc:.2f}%")

print("\nSaving DAPO fine-tuned model...")
model.save_pretrained("dapo_finetuned_model")
tokenizer.save_pretrained("dapo_finetuned_model")
print("Done.")
