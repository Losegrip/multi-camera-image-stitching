"""
illumination.py

实现两类光照补偿方法：
1. 全局比例光照补偿 (compute_global_gain_scalar)
2. 基于网格增益 surface 的局部光照补偿 (compute_grid_gain_surface)

两种方法都基于参考图 / 目标图在重合区域的亮度统计，
输出为用于逐像素缩放的增益因子或增益 surface。
"""
import numpy as np
import cv2


GRID_SIZE_DEFAULT       = (8, 8)
CLIP_RANGE_DEFAULT      = (0.5, 2.0)
DELTA_THR_DEFAULT       = 0.03
ROBUST_PERCENT_DEFAULT  = 5.0
ALPHA_SOFT_DEFAULT      = 0.6

# ---------------------------------------------------------
# 计算重合区域
# ---------------------------------------------------------
def _compute_overlap(mask_ref: np.ndarray, mask_tgt: np.ndarray):
    overlap = (mask_ref > 0) & (mask_tgt > 0)
    if not np.any(overlap):
        return None, overlap
    ys, xs = np.where(overlap)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return (x_min, y_min, x_max, y_max), overlap

# ---------------------------------------------------------
# 整体比例光照补偿法
# ---------------------------------------------------------
def compute_global_gain_scalar(
    img_ref: np.ndarray,
    img_tgt: np.ndarray,
    overlap_mask: np.ndarray,
    clip_range=(0.5, 2.0),
    eps: float = 1e-3,
) -> float:
    # 灰度
    if img_ref.ndim == 3:
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        gray_tgt = cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY)
    else:
        gray_ref = img_ref
        gray_tgt = img_tgt

    ref_pix = gray_ref[overlap_mask].astype(np.float32)
    tgt_pix = gray_tgt[overlap_mask].astype(np.float32)

    if ref_pix.size == 0 or tgt_pix.size == 0:
        return 1.0

    ag1 = ref_pix.mean()
    ag2 = tgt_pix.mean()

    g = ag1 / (ag2 + eps)
    g = float(np.clip(g, clip_range[0], clip_range[1]))
    return g

def apply_gain_map_color(img_tgt: np.ndarray, gain_map: np.ndarray) -> np.ndarray:
    """
    对 BGR 彩色图应用 gain_map（H x W x 1 或 H x W x 3）。

    参数
    ----
    img_tgt : uint8 BGR 图像
    gain_map : float32 增益图，值通常在 [0.5, 2.0] 范围内，
               shape 为 (H, W, 1) 或 (H, W, 3)。

    返回
    ----
    out : uint8 BGR 图像，已经做了增益缩放和 [0, 255] 截断。
    """
    img_f = img_tgt.astype(np.float32)
    if gain_map.shape[2] == 1:
        gain_map = np.repeat(gain_map, 3, axis=2)

    out = img_f * gain_map
    out = np.clip(out, 0, 255)
    return out.astype(img_tgt.dtype)
# ---------------------------------------------------------
# 本文改进光照补偿法
# ---------------------------------------------------------
def compute_grid_gain_surface(
    img_ref: np.ndarray,
    img_tgt: np.ndarray,
    mask_ref: np.ndarray,
    mask_tgt: np.ndarray,
    grid_size= GRID_SIZE_DEFAULT,
    clip_range= CLIP_RANGE_DEFAULT,   
    eps: float = 1e-3,
    delta_thr: float = DELTA_THR_DEFAULT,
    robust_percent: float = ROBUST_PERCENT_DEFAULT, 
) -> np.ndarray:
    """
    网格增益 Surface 光照补偿方法（本工程改进法）：

    1. 在重合区域内划分 grid_h x grid_w 个网格；
    2. 每个网格单独估计局部增益 g_ij（鲁棒均值 + 限幅）；
    3. 对 g_ij 做双线性插值，得到平滑的增益面 G(x,y)；
    4. 在重合区域应用 G(x,y)，其它区域增益=1。

    返回:
        gain_map: (H, W, 1) float32
    """
    h, w = img_ref.shape[:2]
    bbox, overlap = _compute_overlap(mask_ref, mask_tgt)
    if bbox is None:
        # 没有重合区域，直接返回全 1
        return np.ones((h, w, 1), dtype=np.float32)

    x_min, y_min, x_max, y_max = bbox
    ov_w = x_max - x_min + 1
    ov_h = y_max - y_min + 1

    grid_h, grid_w = grid_size

    # --- 转灰度（用 luminance 近似） ---
    if img_ref.ndim == 3:
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        gray_tgt = cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY)
    else:
        gray_ref = img_ref
        gray_tgt = img_tgt

    overlap_mask = overlap

    # 存每个网格的局部增益 g_ij
    grid_g = np.ones((grid_h, grid_w), dtype=np.float32)

    for gy in range(grid_h):
        # 当前网格在重合区域内的 y 范围（整除划分）
        y0 = y_min + gy * ov_h // grid_h
        y1 = y_min + (gy + 1) * ov_h // grid_h
        if gy == grid_h - 1:
            y1 = y_max + 1

        for gx in range(grid_w):
            # 当前网格在重合区域内的 x 范围
            x0 = x_min + gx * ov_w // grid_w
            x1 = x_min + (gx + 1) * ov_w // grid_w
            if gx == grid_w - 1:
                x1 = x_max + 1

            cell_mask = overlap_mask[y0:y1, x0:x1]
            if not np.any(cell_mask):
                # 这个网格没有重合像素，增益保持 1
                g_cell = 1.0
            else:
                ref_pix = gray_ref[y0:y1, x0:x1][cell_mask].astype(np.float32)
                tgt_pix = gray_tgt[y0:y1, x0:x1][cell_mask].astype(np.float32)

                if ref_pix.size < 10:
                    g_cell = 1.0
                else:
                    # --- 去掉最亮/最暗的一部分像素，防高光/阴影干扰 ---
                    if robust_percent > 0:
                        lo = np.percentile(ref_pix, robust_percent)
                        hi = np.percentile(ref_pix, 100.0 - robust_percent)
                        mid_mask = (ref_pix >= lo) & (ref_pix <= hi)
                        if mid_mask.sum() > 10:
                            ref_use = ref_pix[mid_mask]
                            tgt_use = tgt_pix[mid_mask]
                        else:
                            ref_use = ref_pix
                            tgt_use = tgt_pix
                    else:
                        ref_use = ref_pix
                        tgt_use = tgt_pix

                    mu_ref = float(ref_use.mean())
                    mu_tgt = float(tgt_use.mean())
                    if mu_tgt < eps:
                        g_cell = 1.0
                    else:
                        g_raw = mu_ref / (mu_tgt + eps)

                        # 若本地差距很小，则认为不用补偿
                        if abs(g_raw - 1.0) < delta_thr:
                            g_cell = 1.0
                        else:
                            # 只拉一部分差距，避免过激
                            alpha = 0.6  # 0~1，越小越温和
                            g_soft = 1.0 + alpha * (g_raw - 1.0)
                            g_cell = float(
                                np.clip(g_soft, clip_range[0], clip_range[1])
                            )

            grid_g[gy, gx] = g_cell

    # --- 用双线性插值把 grid_g 放大到重合区域大小 (ov_h, ov_w) ---
    gain_overlap = cv2.resize(
        grid_g,
        (ov_w, ov_h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    # --- 嵌回整幅画布，重合区用 gain_overlap，其余为 1 ---
    gain_map = np.ones((h, w, 1), dtype=np.float32)
    gain_map[y_min:y_max + 1, x_min:x_max + 1, 0] = gain_overlap

    return gain_map
