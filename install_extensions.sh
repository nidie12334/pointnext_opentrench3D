#!/usr/bin/env bash
set -euo pipefail

# —— 配置 —— #
# 1. 如果你已经在交互式 shell 中初始化了 Conda，这里可以直接激活：
#    否则需要先 `source` Conda 脚本，比如：
# source ~/anaconda3/etc/profile.d/conda.sh
# 修改下面这行，使之与你的 Conda 安装位置和环境名一致：


# —— 开始安装 —— #
echo "==> 编译并安装 PointNet++ 扩展"
pushd openpoints/cpp/pointnet2_batch >/dev/null
python setup.py install
popd >/dev/null

echo "==> 编译 Grid Subsampling（可选）"
pushd openpoints/subsampling >/dev/null
python setup.py build_ext --inplace
popd >/dev/null

echo "==> 安装 PointOps（Point Transformer 等）"
pushd openpoints/pointops >/dev/null
python setup.py install
popd >/dev/null

echo "==> 安装 Chamfer 距离（可选，重建任务）"
pushd openpoints/chamfer_dist >/dev/null
python setup.py install --user
popd >/dev/null

echo "==> 安装 EMD（可选，重建任务）"
pushd openpoints/emd >/dev/null
python setup.py install --user
popd >/dev/null

echo "✅ 所有 C++ 扩展安装完成！"
