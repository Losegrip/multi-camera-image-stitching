from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def cylindrical_project(
    image: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对输入图像进行柱面投影（绕竖直轴），返回柱面展开图和映射表。

    参数
    ----
    image : np.ndarray
        原始图像，H x W x 3 (uint8)。
    fx, fy : float
        原始相机在 x/y 方向的焦距（像素单位）。
        没有标定时可以先用 W 或 (W+H)/2 近似。
    cx, cy : float
        原始相机主点坐标（像素），一般是 (W/2, H/2)。

    返回
    ----
    cyl : np.ndarray
        柱面展开后的图像，尺寸与输入图像相同 (H, W)。
    map_xy : np.ndarray
        H x W x 2 的 float32 映射表，
        map_xy[y, x] = (u, v) 表示柱面图该像素来自原图 (u, v)。
        方便以后需要从柱面坐标反推回原始像平面时使用。
    """
    if image.ndim != 3:
        raise ValueError(f"Expect 3-channel image, got shape {image.shape}")

    h, w = image.shape[:2]

    # 输出柱面图大小目前就先取和输入一致，方便对比
    out_h, out_w = h, w

    # 构建输出图每个像素的网格坐标
    ys, xs = np.indices((out_h, out_w), dtype=np.float32)

    # 以输出图中心为原点的坐标（单位：像素）
    x_c = xs - out_w * 0.5
    y_c = ys - out_h * 0.5

    # 圆柱半径取 R = fx（常见做法）
    R = fx

    # 对应的圆柱角度和高度
    # θ: 沿水平方向的视角（弧度），h_cyl: 在圆柱上的相对高度
    theta = x_c / R           # [-?, ?]
    h_cyl = y_c / R           # 近似做法：用相同尺度

    # 从圆柱坐标反投到相机归一化平面
    # 圆柱参数化： (X, Y, Z) = (sinθ, h, cosθ)
    X = np.sin(theta)
    Z = np.cos(theta)
    Y = h_cyl

    # 避免除零
    Z[Z == 0] = 1e-8

    x_n = X / Z
    y_n = Y / Z

    # 再从相机平面投到原始像素坐标
    u = fx * x_n + cx
    v = fy * y_n + cy

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    map_xy = np.stack([map_x, map_y], axis=-1)

    # remap 采样生成柱面展开图
    cyl = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return cyl, map_xy
