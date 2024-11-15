import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_tensor_3d(array_hwc: np.array) -> Axes3D:
    # 确保tensor是numpy数组
    array_hwc = np.asarray(array_hwc)

    # 获取张量的维度
    dim_x, dim_y, dim_z = array_hwc.shape

    # 创建网格数据
    X = np.linspace(0, 1, dim_x)
    Y = np.linspace(0, 1, dim_y)
    Z = np.linspace(0, 1, dim_z)

    # 创建网格
    x, y, z = np.meshgrid(X, Y, Z, indexing="ij")

    # 创建一个新的图形
    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    ax.set_alpha(0)  # 设置背景透明

    # 绘制xy平面 (Z固定)
    Z_fixed = np.ones((dim_x, dim_y))
    ax.plot_surface(x[:, :, 0], y[:, :, 0], Z_fixed, facecolors=plt.cm.viridis(array_hwc[:, :, 0]), shade=False, alpha=1)

    # 绘制xz平面 (Y固定)
    Y_fixed = np.zeros((dim_x, dim_z))
    ax.plot_surface(x[:, 0, :], Y_fixed, z[:, 0, :], facecolors=plt.cm.viridis(array_hwc[:, 0, :]), shade=False, alpha=1)

    # 绘制yz平面 (X固定)
    X_fixed = np.ones((dim_y, dim_z))
    ax.plot_surface(X_fixed, y[0, :, :], z[0, :, :], facecolors=plt.cm.viridis(array_hwc[0, :, :]), shade=False, alpha=1)
    ax.set_axis_off()
    return ax
