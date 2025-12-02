import numpy as np
import cv2


def render_pano_cover(world_warp_list, world_masks):
    """粗暴覆盖版，全局画布 baseline。"""
    assert len(world_warp_list) == len(world_masks)
    h, w = world_warp_list[0].shape[:2]

    pano = np.zeros((h, w, 3), dtype=np.uint8)
    pano_mask = np.zeros((h, w), dtype=bool)

    for img, mask in zip(world_warp_list, world_masks):
        mask = mask.astype(bool)
        pano[mask] = img[mask]
        pano_mask |= mask

    return pano, pano_mask


def _build_soft_window(mask: np.ndarray, blur_ksize: int = 15) -> np.ndarray:
    """
    根据有效掩膜生成平滑权重窗：中心高、边缘低。
    使用距离变换 + 高斯平滑，归一化到 [0,1]，掩膜外为 0。
    """
    mask_u8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    if dist.max() > 0:
        dist = dist / dist.max()
    if blur_ksize and blur_ksize > 1:
        dist = cv2.GaussianBlur(dist, (blur_ksize, blur_ksize), 0)
    window = dist * (mask_u8 > 0)
    return window.astype(np.float32)


def render_pano_feather(world_warp_list, world_masks, ref_id=0, feather_width=32):
    """
    全图渐变窗融合：每路图像使用平滑权重窗，叠加后归一化。
    """
    assert len(world_warp_list) == len(world_masks)
    h, w = world_warp_list[0].shape[:2]

    pano_num = np.zeros((h, w, 3), dtype=np.float32)
    pano_den = np.zeros((h, w), dtype=np.float32)

    for img, mask in zip(world_warp_list, world_masks):
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            continue
        win = _build_soft_window(mask_bool)
        pano_num += img.astype(np.float32) * win[..., None]
        pano_den += win

    pano = np.zeros_like(world_warp_list[0])
    valid = pano_den > 1e-6
    pano[valid] = (pano_num[valid] / pano_den[valid, None])
    pano = np.clip(pano, 0, 255).astype(np.uint8)

    return pano, (pano_den > 1e-6)
