#!/usr/bin/env python3
import os, glob
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from tqdm import tqdm

# —— 配置区 —— 
# 原始数据所在目录
DIRS = [
    "/home/tech/pointnext/datasets/OpenTrench3D/water/train",
    "/home/tech/pointnext/datasets/OpenTrench3D/water/test"
]
# 预处理后输出目录后缀
OUT_SUFFIX = "_pre"
# 邻域大小（KNN）用于法向量和曲率估计
KNN = 30
# —— end 配置 —— 

def ensure_outdir(in_dir):
    parent, name = os.path.split(in_dir.rstrip('/'))
    out_dir = os.path.join(parent, name + OUT_SUFFIX)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def process_ply(ply_path, out_dir):
    # 1. 读取原始 PLY 顶点数据
    ply = PlyData.read(ply_path)
    vert = ply['vertex']
    x = vert['x']; y = vert['y']; z = vert['z']
    r = vert['red']; g = vert['green']; b = vert['blue']
    cls = vert['class']  # uchar

    # 2. 构建 Open3D 点云并估计法向量
    pts = np.stack([x, y, z], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=KNN)
    )
    normals = np.asarray(pcd.normals)  # (N,3)

    # 3. 基于邻域 PCA 估计曲率
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    curv = np.zeros(len(pts), dtype=np.float32)
    for i in tqdm(range(len(pts)), desc=f"Curvature {os.path.basename(ply_path)}"):
        _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], KNN)
        nbr = pts[idx, :]
        cov = np.cov(nbr.T)
        eigs, _ = np.linalg.eigh(cov)
        curv[i] = eigs[0] / (eigs.sum() + 1e-6)

    # 4. 合并属性并写出新的 PLY
    vertex_all = np.empty(len(pts), dtype=[
        ('x','f4'),('y','f4'),('z','f4'),
        ('red','u1'),('green','u1'),('blue','u1'),
        ('nx','f4'),('ny','f4'),('nz','f4'),
        ('curvature','f4'),
        ('class','u1'),
    ])
    vertex_all['x'], vertex_all['y'], vertex_all['z'] = x, y, z
    vertex_all['red'], vertex_all['green'], vertex_all['blue'] = r, g, b
    vertex_all['nx'], vertex_all['ny'], vertex_all['nz'] = normals.T
    vertex_all['curvature'] = curv
    vertex_all['class'] = cls

    out_path = os.path.join(out_dir, os.path.basename(ply_path).replace('.ply','_geom.ply'))
    PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True).write(out_path)

def main():
    for in_dir in DIRS:
        out_dir = ensure_outdir(in_dir)
        files = sorted(glob.glob(os.path.join(in_dir, '*.ply')))
        if not files:
            print(f"[WARN] 目录 {in_dir} 下无 PLY 文件，跳过")
            continue
        print(f"Processing: {in_dir} → {out_dir}, {len(files)} files")
        for ply in files:
            process_ply(ply, out_dir)
    print("预处理完成！")

if __name__ == "__main__":
    main()
