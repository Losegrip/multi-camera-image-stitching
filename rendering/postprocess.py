# 拼接结果的后处理工具：自动裁剪有效区域 + 统一缩放

from __future__ import annotations

from typing import Tuple, Optional

import cv2
import numpy as np

# (x, y, w, h)
BBox = Tuple[int, int, int, int]

def compute_valid_bbox(
    pano: np.ndarray,
    min_valid_fraction: float = 1e-4,
) -> BBox:
    """
    计算全景图中“非全黑区域”的最小外接矩形。

    参数
    ----
    pano : np.ndarray
        全景图，H x W 或 H x W x C，要求无效区域为全 0。
    min_valid_fraction : float
        如果有效像素比例低于该阈值，认为图像基本为空，
        此时返回整幅图的 bbox 以避免异常。

    返回
    ----
    (x, y, w, h) : BBox
        有效区域的外接矩形。若有效像素太少，则返回 (0, 0, W, H)。
    """
    if pano.ndim == 3:
        # 任一通道非零即认为有效
        valid_mask = np.any(pano != 0, axis=2)
    elif pano.ndim == 2:
        valid_mask = pano != 0
    else:
        raise ValueError(f"Unsupported pano ndim: {pano.ndim}")

    h, w = valid_mask.shape

    num_valid = int(valid_mask.sum())
    if num_valid < min_valid_fraction * valid_mask.size:
        # 有效像素太少，直接返回整图，避免后续索引错误
        return 0, 0, w, h

    # 有有效像素的行、列索引
    ys = np.where(valid_mask.any(axis=1))[0]
    xs = np.where(valid_mask.any(axis=0))[0]

    y_min, y_max = int(ys[0]), int(ys[-1])
    x_min, x_max = int(xs[0]), int(xs[-1])

    bbox_w = x_max - x_min + 1
    bbox_h = y_max - y_min + 1

    return x_min, y_min, bbox_w, bbox_h


def auto_crop_valid(
    pano: np.ndarray,
    margin: int = 10,
    min_valid_fraction: float = 1e-4,
) -> Tuple[np.ndarray, BBox]:
    """
    自动裁剪掉全黑边界区域，返回裁剪后的全景图和 bbox。

    参数
    ----
    pano : np.ndarray
        输入全景图，要求无效区域为全 0。
    margin : int
        在有效区域外额外保留的像素边距，防止裁得太紧。
    min_valid_fraction : float
        见 compute_valid_bbox。

    返回
    ----
    pano_crop : np.ndarray
        裁剪后的全景图。
    bbox : (x, y, w, h)
        裁剪在原图中的位置。
    """
    h, w = pano.shape[:2]
    x, y, bw, bh = compute_valid_bbox(pano, min_valid_fraction=min_valid_fraction)

    # 四周留出 margin
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w, x + bw + margin)
    y1 = min(h, y + bh + margin)

    pano_crop = pano[y0:y1, x0:x1].copy()
    bbox = (x0, y0, x1 - x0, y1 - y0)
    return pano_crop, bbox


def draw_seams_on_pano(pano: np.ndarray,
                       seam_records,
                       color=(0, 0, 255),
                       thickness: int = 1) -> np.ndarray:
    """
    在全景图上叠加绘制所有 pair 的 seam 曲线
    seam_records: List[(bbox, seam_x_roi)]
      bbox: (x_min, y_min, x_max, y_max)
      seam_x_roi: 长度为 H_roi 的一维数组（ROI 内局部 x 坐标）
    """
    pano_dbg = pano.copy()
    H, W = pano_dbg.shape[:2]

    for bbox, seam_x_roi in seam_records:
        x_min, y_min, x_max, y_max = bbox
        H_roi = y_max - y_min + 1

        for yy in range(H_roi):
            y = y_min + yy
            if y < 0 or y >= H:
                continue
            sx = int(seam_x_roi[yy]) + x_min
            if sx < 0 or sx >= W:
                continue

            # 画一个小水平线段比单点更容易看见
            x0 = max(0, sx - 1)
            x1 = min(W - 1, sx + 1)
            cv2.line(pano_dbg, (x0, y), (x1, y), color, thickness)

    return pano_dbg
