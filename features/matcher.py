# features/matcher.py
from typing import List, Dict
import cv2
import numpy as np
from .types import FeatureConfig, ImageFeatures, Match, MatchResult


def _bf_l2_matcher() -> cv2.BFMatcher:
    """创建用于 SIFT 描述子的 L2 距离 BFMatcher。"""
    return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


def match_two_views(
    f1: ImageFeatures,
    f2: ImageFeatures,
    cfg: FeatureConfig,
) -> MatchResult:
    """
    对两幅图做特征匹配：KNN + ratio test (可选 cross-check)。
    """
    if f1.descriptors is None or f2.descriptors is None:
        return MatchResult(f1.image_id, f2.image_id, matches=[])

    bf = _bf_l2_matcher()

    # 1) 正向 knn 匹配
    raw_12 = bf.knnMatch(f1.descriptors, f2.descriptors, k=2)

    def _apply_ratio(knn_matches):
        good = []
        for m, n in knn_matches:
            if m.distance < cfg.ratio_test * n.distance:
                good.append(m)
        return good

    good_12 = _apply_ratio(raw_12)

    if not cfg.cross_check:
        matches = [
            Match(m.queryIdx, m.trainIdx, float(m.distance)) for m in good_12
        ]
        return MatchResult(f1.image_id, f2.image_id, matches)

    raw_21 = bf.knnMatch(f2.descriptors, f1.descriptors, k=2)
    good_21 = _apply_ratio(raw_21)

    # 建立 query -> best train 映射
    best_12: Dict[int, int] = {m.queryIdx: m.trainIdx for m in good_12}
    best_21: Dict[int, int] = {m.queryIdx: m.trainIdx for m in good_21}

    mutual_matches: List[Match] = []
    for q1, t1 in best_12.items():
        if t1 in best_21 and best_21[t1] == q1:
            # 计算两个描述子的 L2 距离
            diff = f1.descriptors[q1] - f2.descriptors[t1]
            d = float(np.linalg.norm(diff))
            mutual_matches.append(Match(q1, t1, d))

    return MatchResult(f1.image_id, f2.image_id, mutual_matches)


def match_chain_sequential(
    features_list: List[ImageFeatures],
    cfg: FeatureConfig,
) -> List[MatchResult]:
    """
    对一串图像按 (0-1, 1-2, 2-3, ...) 链式匹配，适合你的多相机阵列。
    """
    results: List[MatchResult] = []
    for i in range(len(features_list) - 1):
        res = match_two_views(features_list[i], features_list[i + 1], cfg)
        results.append(res)
    return results