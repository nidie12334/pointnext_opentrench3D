# script/stat_class_freq.py
import glob, os
import numpy as np

# 你数据集的训练子集目录
root = '/home/tech/pointnext/datasets/OpenTrench3D/water/train'
counts = np.zeros(5, dtype=int)

for ply_path in glob.glob(os.path.join(root, '*.ply')):
    with open(ply_path, 'r') as f:
        # 跳过 header
        for line in f:
            if line.strip() == 'end_header':
                break
        # 读剩下每行最后一个数字
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[-1])   # 最后一列就是 class
            counts[cls] += 1

print('class counts:', counts)

# 频率 & 权重计算
freq = counts / counts.sum()
weights = 1.0 / np.log(1.02 + freq)
print('class weights:', weights.tolist())
