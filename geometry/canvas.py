import cv2
import numpy as np
from typing import Tuple, List

def compute_panorama_canvas(
    images: List[np.ndarray],
    H_to_ref: List[np.ndarray],  # 每张图 -> 参考坐标系 的 3x3 矩阵
    ref_size: Tuple[int, int],   # (h_ref, w_ref)，比如用参考图尺寸
):
    """
    根据各图像到参考坐标系的单应 H_to_ref，计算“世界画布”（全景画布）。

    参数
    ----
    images : List[np.ndarray]
        原始输入图像列表。
    H_to_ref : List[np.ndarray]
        每张图到“参考视图坐标系”的 3x3 单应矩阵。
    ref_size : (h_ref, w_ref)
        参考视图自身的尺寸，用于确定坐标系尺度。

    返回
    ----
    pano_h, pano_w : int
        世界画布的高度和宽度。
    T : np.ndarray
        3x3 平移单应矩阵，将“参考坐标系”平移到画布左上角为 (0,0) 的坐标系。
        也就是：x_canvas = T @ x_ref
    """
    h_ref, w_ref = ref_size

    # 1. 把每张图的 4 个角投影到 “参考坐标” 下
    all_pts = []
    for img, H in zip(images, H_to_ref):
        h, w = img.shape[:2]
        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1],
        ], dtype=np.float32).T  # 3x4

        pts_ref = H @ corners
        pts_ref /= pts_ref[2:3, :]
        all_pts.append(pts_ref[:2, :])  # 2x4

    all_pts = np.hstack(all_pts)  # 2 x (4*N)

    xs = all_pts[0, :]
    ys = all_pts[1, :]

    # 2. 求出包围所有投影点的最小包围框
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    # 3. 计算画布大小（向上取整）
    pano_w = int(np.ceil(max_x - min_x))
    pano_h = int(np.ceil(max_y - min_y))

    # 4. 计算把“参考坐标”搬到画布坐标的平移 offset
    offset_x = -min_x
    offset_y = -min_y

    T = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1],
    ], dtype=np.float32)

    return pano_h, pano_w, T