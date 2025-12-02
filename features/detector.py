# features/detector.py
from typing import Optional, Tuple, List
import numpy as np
import cv2

from .types import FeatureConfig, ImageFeatures


def _create_sift(cfg: FeatureConfig) -> cv2.SIFT:
    return cv2.SIFT_create()


def _grid_select_keypoints(
    keypoints: List[cv2.KeyPoint],
    descriptors: np.ndarray,
    img_shape: Tuple[int, int],
    cfg: FeatureConfig,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    按网格均匀保留特征点，避免都挤在同一块区域。

    思路：
      1. 把图像分成 grid_rows x grid_cols 个 cell；
      2. 每个 cell 内按 response 排序，取前 max_per_cell 个特征；
      3. 全局如果超过 max_keypoints，再截断一次。
    """
    if not keypoints or descriptors is None:
        return [], descriptors

    h, w = img_shape
    R, C = cfg.grid_rows, cfg.grid_cols
    max_per_cell = max(1, cfg.max_keypoints // (R * C))

    cells: List[List[Tuple[float, int]]] = [
        [] for _ in range(R * C)
    ]

    for idx, kp in enumerate(keypoints):
        x, y = kp.pt
        c = min(int(x / w * C), C - 1)
        r = min(int(y / h * R), R - 1)
        cell_idx = r * C + c
        cells[cell_idx].append((kp.response, idx))

    selected_indices: List[int] = []
    for cell in cells:
        if not cell:
            continue
        cell_sorted = sorted(cell, key=lambda t: t[0], reverse=True)
        selected_indices.extend(idx for _, idx in cell_sorted[:max_per_cell])

    if len(selected_indices) > cfg.max_keypoints:
        selected_indices = sorted(
            selected_indices,
            key=lambda i: keypoints[i].response,
            reverse=True,
        )[: cfg.max_keypoints]

    selected_indices = sorted(set(selected_indices))

    selected_kps = [keypoints[i] for i in selected_indices]
    selected_desc = descriptors[selected_indices, :] if descriptors is not None else None
    return selected_kps, selected_desc


def extract_sift_features(
    image_bgr: np.ndarray,
    image_id: int,
    cfg: FeatureConfig,
    mask: Optional[np.ndarray] = None,
) -> ImageFeatures:
    """
    输入 BGR 或灰度图，输出一张图的 SIFT 特征。

    流程：
      1. 转灰度；
      2. 用 SIFT 检测 + 计算描述子；
      3. 用网格做一次均匀采样控制特征数量；
      4. 打包成 ImageFeatures 返回。

    参数:
      image_bgr: 原图 (H, W, 3) 或灰度 (H, W)
      image_id: 这一张图的编号（比如相机 index）
      cfg: FeatureConfig
      mask: 可选掩膜（为 0 的地方不检测特征，例如黑边 / 无效区域）
    """
    assert image_bgr.ndim in (2, 3), "image must be gray or BGR"

    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr

    h, w = gray.shape[:2]

    sift = _create_sift(cfg)
    keypoints, descriptors = sift.detectAndCompute(gray, mask)

    if keypoints and descriptors is not None:
        keypoints, descriptors = _grid_select_keypoints(
            keypoints, descriptors, (h, w), cfg
        )
    else:
        keypoints, descriptors = [], None

    return ImageFeatures(
        image_id=image_id,
        shape=(h, w),
        keypoints=keypoints,
        descriptors=descriptors,
    )