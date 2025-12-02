from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2

from features.types import ImageFeatures, Match, MatchResult


@dataclass
class HomographyResult:
    """一对图像之间的单应估计结果，表示从 img_id1 到 img_id2 的单应 H。"""
    img_id1: int
    img_id2: int
    H: np.ndarray | None              # 3x3 单应矩阵，失败时为 None
    inlier_mask: np.ndarray | None    # (N, ) bool，指示哪些匹配是内点
    num_inliers: int                  # 内点数量
    num_matches: int                  # 总匹配数量

    @property
    def inlier_ratio(self) -> float:
        if self.num_matches == 0:
            return 0.0
        return self.num_inliers / float(self.num_matches)


def _matches_to_points(
    f1: ImageFeatures,
    f2: ImageFeatures,
    matches: List[Match],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 Match 列表转换成两个点集：
      pts1[i] <-> pts2[i]
    """
    pts1 = []
    pts2 = []
    for m in matches:
        kp1 = f1.keypoints[m.query_idx]
        kp2 = f2.keypoints[m.train_idx]
        pts1.append(kp1.pt)
        pts2.append(kp2.pt)

    if len(pts1) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    return pts1, pts2


def estimate_homography_ransac(
    f1: ImageFeatures,
    f2: ImageFeatures,
    match_result: MatchResult,
    ransac_thresh: float = 3.0,
    confidence: float = 0.995,
) -> HomographyResult:
    """
    使用 RANSAC 从匹配中估计单应矩阵 H。

    输入:
      f1, f2: 两张图像的特征
      match_result: 两图之间的匹配结果（已经过 ratio test 等）
      ransac_thresh: RANSAC 重投影误差阈值（像素）
      confidence: 置信度（OpenCV 会据此决定迭代次数）

    输出:
      HomographyResult, 包含:
        - H: 3x3 单应矩阵（或 None）
        - inlier_mask: (N,) bool 数组，True 表示该匹配为内点
        - num_inliers / num_matches
    """
    matches = match_result.matches
    num_matches = len(matches)

    if num_matches < 4:
        # 单应至少需要 4 个对应点
        return HomographyResult(
            img_id1=match_result.img_id1,
            img_id2=match_result.img_id2,
            H=None,
            inlier_mask=None,
            num_inliers=0,
            num_matches=num_matches,
        )

    pts1, pts2 = _matches_to_points(f1, f2, matches)

    # OpenCV RANSAC 单应估计
    # 注意：Python 接口中，第 4 个参数就是 ransacReprojThreshold
    H, mask = cv2.findHomography(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=confidence,
    )

    if H is None or mask is None:
        return HomographyResult(
            img_id1=match_result.img_id1,
            img_id2=match_result.img_id2,
            H=None,
            inlier_mask=None,
            num_inliers=0,
            num_matches=num_matches,
        )

    # mask 是 (N, 1) 的 0/1 数组，这里转成 (N,) bool
    mask = mask.ravel().astype(bool)
    num_inliers = int(mask.sum())

    return HomographyResult(
        img_id1=match_result.img_id1,
        img_id2=match_result.img_id2,
        H=H,
        inlier_mask=mask,
        num_inliers=num_inliers,
        num_matches=num_matches,
    )