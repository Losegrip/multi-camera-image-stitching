# geometry/warp.py
from typing import Tuple, Optional
import numpy as np
import cv2

from views.types import View


def warp_view_to_ref(
    view: View,
    ref_shape: Tuple[int, int],
) -> np.ndarray:
    """
    将选择的一路视图 warp 到参考视图坐标系。

    Parameters
    ----------
    view : View
        含有 H_to_ref 的视图。
    ref_shape : (h_ref, w_ref)
        参考视图的高宽，warp 输出也会用这个尺寸。

    Returns
    -------
    warped : np.ndarray
        已经对齐到参考坐标系的图像。
    """
    if view.H_to_ref is None:
        raise ValueError(f"View {view.id} has no H_to_ref; cannot warp to reference.")

    H = view.H_to_ref
    h_ref, w_ref = ref_shape

    warped = cv2.warpPerspective(view.image, H, (w_ref, h_ref))
    return warped

def warp_image_and_mask(
    img: np.ndarray,
    H_canvas: np.ndarray,
    out_shape: Tuple[int, int],
    src_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将图像和对应的有效区域 mask 一起 warp 到世界画布。

    参数
    ----
    img : np.ndarray
        输入图像（灰度或 3 通道）。
    H_canvas : np.ndarray
        3x3 单应矩阵，将图像坐标映射到世界画布坐标。
    out_shape : (h_out, w_out)
        输出画布的高度和宽度。
    src_mask : Optional[np.ndarray]
        源图像的有效区域 mask（非零为有效）。
        若为 None，则自动以“非全零像素”为有效区域。

    返回
    ----
    warped : np.ndarray
        warp 后的图像。
    valid : np.ndarray
        bool 掩膜，表示输出画布上哪些位置有来自该图像的有效像素。
    """
    h_out, w_out = out_shape
    warped = cv2.warpPerspective(
        img, H_canvas, (w_out, h_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    if src_mask is None:
        if img.ndim == 3:
            src_mask = (img.sum(axis=2) > 0).astype(np.uint8)
        else:
            src_mask = (img > 0).astype(np.uint8)
    else:
        if src_mask.ndim == 3:
            src_mask = src_mask[..., 0]
        src_mask = (src_mask > 0).astype(np.uint8)

    warped_mask = cv2.warpPerspective(
        src_mask, H_canvas, (w_out, h_out),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    warped_mask = cv2.erode(warped_mask, np.ones((3,3),np.uint8), iterations=1)
    valid = warped_mask > 0
    return warped, valid
