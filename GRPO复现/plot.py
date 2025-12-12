import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ----------------------
# 1. 数据准备
# ----------------------
data = {
    '阶段': ['GRPO训练前', 'GRPO训练后'],
    '正确率': [16.67, 76.67],
    '答对题数': [5, 23],
    '总题数': [30, 30]
}

df = pd.DataFrame(data)

# ----------------------
# 2. 画布与风格设置
# ----------------------
# 设置绘图风格为简洁的白底网格
sns.set_theme(style="whitegrid")

# 设置中文字体，防止乱码 (根据系统不同，可能需要调整字体名称)
# Windows常用 SimHei, Mac常用 Arial Unicode MS
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

# 创建画布大小
plt.figure(figsize=(8, 6), dpi=100)

# ----------------------
# 3. 绘制柱状图
# ----------------------
# 定义颜色：灰色代表训练前（平淡），绿色代表训练后（成功/提升）
colors = ['#bdc3c7', '#2ecc71'] 

# 绘制条形图
ax = sns.barplot(x='阶段', y='正确率', data=df, palette=colors, hue='阶段', legend=False)

# ----------------------
# 4. 添加美化细节与标注
# ----------------------
# 遍历每个柱子，添加具体的数值标签
for index, row in df.iterrows():
    # 获取柱子对象
    bar = ax.patches[index]
    # 获取柱子高度和位置
    height = bar.get_height()
    x_pos = bar.get_x() + bar.get_width() / 2
    
    # 1. 在柱子上方显示百分比
    ax.text(x_pos, height + 1.5, f'{row["正确率"]}%', 
            ha='center', va='bottom', fontsize=14, fontweight='bold', color='#34495e')
    
    # 2. 在柱子内部显示具体题数 (5/30)
    ax.text(x_pos, height/2, f'({row["答对题数"]}/{row["总题数"]})', 
            ha='center', va='center', fontsize=11, color='white', fontweight='bold')

# 设置标题和轴标签
plt.title('GRPO 训练前后模型评估对比', fontsize=18, pad=20, fontweight='bold')
plt.ylabel('正确率 (%)', fontsize=12)
plt.xlabel('') # x轴标签已足够清晰，隐藏轴标题

# 设置Y轴范围，让顶部留出空间给标签
plt.ylim(0, 100)

# 去除顶部和右侧的边框，使图表更现代
sns.despine()

# ----------------------
# 5. 显示图表
# ----------------------
plt.tight_layout()
plt.show()