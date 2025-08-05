import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 读取点云
pcd = o3d.io.read_point_cloud("/home/tech/pointnext/datasets/OpenTrench3D/water/train_pre/Area_5_Site_41_geom.ply")
pts = np.asarray(pcd.points)
norms = np.asarray(pcd.normals)

# 投影到 XY 平面，显示法向量在平面上的分量
plt.figure(figsize=(6,6))
plt.quiver(
    pts[:,0], pts[:,1],      # 点的位置
    norms[:,0], norms[:,1],  # 法向的 XY 分量
    angles='xy', scale_units='xy', scale=1, width=0.002
)
plt.scatter(pts[:,0], pts[:,1], s=1, c='k')
plt.title("XY 平面上的法向量投影")
plt.xlabel("X"); plt.ylabel("Y")
plt.axis('equal')
plt.show()
