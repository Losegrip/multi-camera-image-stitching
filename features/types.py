# features/types.py
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
@dataclass
class FeatureConfig:
    """
    SIFT 特征 & 投影配置

    - max_keypoints: 全图最多保留多少特征点
    - grid_rows, grid_cols: 网格划分，用于均匀采样
    - ratio_test, cross_check: 匹配阶段的策略
    - use_cylindrical, cyl_f_ratio: 是否使用柱面投影及焦距比例
    - fx, fy, cx, cy: 可选的真实相机内参标定（如有则可覆盖默认）
    """
    max_keypoints: int = 2000
    grid_rows: int = 4
    grid_cols: int = 8

    ratio_test: float = 0.75
    cross_check: bool = False
    # 柱面投影 / CP-SIFT 相关配置 =========
    use_cylindrical: bool = False  # 开关
    cyl_f_ratio: float = 0.8       #fx = fy = cyl_f_ratio *

    #有真实的相机内参和畸变系数标定（可改用）
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None

@dataclass
class ImageFeatures:
    image_id: int # 相机/图像编号
    shape: Tuple[int, int] #(H, W)
    keypoints: List[cv2.KeyPoint] #OpenCV keypoint列表
    descriptors: Optional[np.ndarray] #(N, 128) 为空的话代表没有特征

@dataclass
class Match:
    query_idx: int
    train_idx: int
    distance: float

@dataclass
class MatchResult:
    img_id1: int
    img_id2: int
    matches: List[Match]