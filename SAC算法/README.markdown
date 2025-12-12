***环境依赖***
创建虚拟环境
-python 3.12
-gym 0.25.2
-numpy 1.26.4
-gym-super-mario-bros 7.4.0
-pytorch 适配cuda版本
-opencv-python 4.8.1

***文件说明***
-SAC-super-mario.py 算法实现和模型训练玩超级马里奥游戏
-SAC-plot.py 绘制训练过程中的曲线图
-SAC_test.py 测试模型效果

***注意***
该脚本在训练时只训练20万步，这对于agent玩超级马里奥游戏来说比较少，建议训练50万步-1百万步以上才可能达到比较好的效果。