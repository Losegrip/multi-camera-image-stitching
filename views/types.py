from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

from features.types import ImageFeatures

@dataclass
class View:
    id: int                        # 视图编号（0,1,2等）
    name: str                      # 文件名
    image: np.ndarray              # 原图
    features: ImageFeatures        # 特征
    H_to_ref: Optional[np.ndarray] = None  # 本视图 -> 参考视图 的 3x3 Homography