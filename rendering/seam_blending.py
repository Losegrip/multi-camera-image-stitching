import cv2
import numpy as np

def compute_energy_Ec_Eg(
    img_ref, img_tgt, overlap, bbox,
    w_c=1.0, w_g=2.0,
    forbid_ref=None, forbid_tgt=None,
    w_f=10.0,                          
):
    """
    阶段 A：E = w_c * Ec + w_g * Eg，再乘一个纹理权重
      Ec: 颜色差 |I1 - I2|（BGR 三通道平均 L1）
      Eg: 灰度差的梯度差 (Sobel(diff_gray))^2
    只在 overlap 区域内归一化，非 overlap 位置不参与 seam。
    """
    x_min, y_min, x_max, y_max = bbox
    # 注意 bbox 这里 x_max, y_max 是包含的索引，切片时要 +1
    roi_ref = img_ref[y_min:y_max+1, x_min:x_max+1]
    roi_tgt = img_tgt[y_min:y_max+1, x_min:x_max+1]

    h_roi, w_roi = roi_ref.shape[:2]

    # ---------- 1. 颜色差 Ec：用 BGR 三通道 ----------
    ref_f = roi_ref.astype(np.float32) / 255.0  # HxWx3
    tgt_f = roi_tgt.astype(np.float32) / 255.0  # HxWx3

    diff_rgb = ref_f - tgt_f                    # HxWx3
    Ec = np.mean(np.abs(diff_rgb), axis=2)      # HxW，颜色差

    # ---------- 2. 灰度差 + 梯度差 Eg ----------
    gray_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_tgt = cv2.cvtColor(roi_tgt, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    diff_gray = gray_ref - gray_tgt

    gx = cv2.Sobel(diff_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(diff_gray, cv2.CV_32F, 0, 1, ksize=3)
    Eg = gx * gx + gy * gy                      # HxW

    # ---------- 3. overlap 子区域内归一化 ----------
    ov_roi = overlap[y_min:y_max+1, x_min:x_max+1]

    def norm_in_mask(E):
        vals = E[ov_roi]
        if vals.size == 0:
            return np.zeros_like(E, dtype=np.float32)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax <= vmin + 1e-6:
            return np.zeros_like(E, dtype=np.float32)
        En = np.zeros_like(E, dtype=np.float32)
        En[ov_roi] = (E[ov_roi] - vmin) / (vmax - vmin)
        return En

    Ec_n = norm_in_mask(Ec)
    Eg_n = norm_in_mask(Eg)

    # ---------- 4. 纹理权重：鼓励 seam 走高纹理区域 ----------
    # 用 ref 灰度的梯度幅值做一个纹理强度
    gx_r = cv2.Sobel(gray_ref, cv2.CV_32F, 1, 0, ksize=3)
    gy_r = cv2.Sobel(gray_ref, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx_r * gx_r + gy_r * gy_r)  # HxW

    texture = np.zeros_like(grad_mag, dtype=np.float32)
    vals_tex = grad_mag[ov_roi]
    if vals_tex.size > 0:
        tmin, tmax = float(vals_tex.min()), float(vals_tex.max())
        if tmax > tmin + 1e-6:
            texture[ov_roi] = (vals_tex - tmin) / (tmax - tmin)
    # 纹理越强，权重越小（能量打折），范围 [0.5, 1.0]
    alpha = 0.5  # 0.3~0.7
    texture_w = 1.0 - alpha * texture           # HxW

    # ---------- 5. 合成最终能量 ----------
    E_raw = w_c * Ec_n + w_g * Eg_n
    E = E_raw * texture_w

    # ---------- 6. 掩膜惩罚 Ef：尽量不穿过关键目标 ----------
    if forbid_ref is not None or forbid_tgt is not None:
        Ef = np.zeros_like(E, dtype=np.float32)

        if forbid_ref is not None:
            fr_roi = forbid_ref[y_min:y_max+1, x_min:x_max+1] > 0
            Ef[fr_roi] = 1.0

        if forbid_tgt is not None:
            ft_roi = forbid_tgt[y_min:y_max+1, x_min:x_max+1] > 0
            Ef[ft_roi] = 1.0

        Ef[~ov_roi] = 0.0
        E = E + w_f * Ef

    # overlap 位置设为大值，禁止接缝线走
    big = 1e3
    E[~ov_roi] = big
    return E

def compute_overlap_and_bbox(mask_ref, mask_tgt):
    """
    输入：两张掩膜 HxW，>0 表示有效像素
    输出：
      overlap: bool HxW，重叠区域
      bbox: (x_min, y_min, x_max, y_max)，如果无重叠返回 None
    """
    overlap = (mask_ref > 0) & (mask_tgt > 0)
    ys, xs = np.where(overlap)
    if len(xs) == 0:
        return overlap, None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return overlap, (int(x_min), int(y_min), int(x_max), int(y_max))

def find_vertical_seam_dp(E: np.ndarray) -> np.ndarray:
    """
    向量化版本：对每一行只循环 y，一行内的转移用 NumPy 处理
    E: H x W
    return: seam_x_roi, shape (H,)
    """
    H, W = E.shape
    dp   = np.zeros_like(E, dtype=np.float32)
    back = np.zeros_like(E, dtype=np.int32)

    dp[0] = E[0]

    cols = np.arange(W, dtype=np.int32)

    for y in range(1, H):
        prev = dp[y - 1]  # (W,)

        # 三个方向的累计代价：左上 / 正上 / 右上
        left  = np.empty_like(prev)
        mid   = prev
        right = np.empty_like(prev)

        # 左边界：没有更左的，只能来自自己
        left[0]  = prev[0]
        left[1:] = prev[:-1]

        # 右边界：没有更右的，只能来自自己
        right[:-1] = prev[1:]
        right[-1]  = prev[-1]

        stack = np.stack([left, mid, right], axis=0)  # (3, W)

        # 对每一列，三种方向中选最小
        idx_dir = np.argmin(stack, axis=0)            # (W,), 取值 0/1/2

        prev_min = stack[idx_dir, cols]               # 选中的最小值

        dp[y] = E[y] + prev_min

        # 记录回溯列坐标
        back_y = cols.copy()
        back_y[idx_dir == 0] -= 1
        # idx_dir == 1 保持不变
        back_y[idx_dir == 2] += 1
        # 边界裁剪
        back_y[back_y < 0]   = 0
        back_y[back_y >= W]  = W - 1
        back[y] = back_y

    seam_x = np.zeros(H, dtype=np.int32)
    seam_x[-1] = int(np.argmin(dp[-1]))
    for y in range(H - 2, -1, -1):
        seam_x[y] = back[y + 1, seam_x[y + 1]]

    return seam_x

def build_pair_weights_from_seam(mask_ref, mask_tgt,
                                 overlap, bbox, seam_x_roi,
                                 band_width=16):
    """
    输出:
      w_ref_pair, w_tgt_pair: HxW float32, 只在这个 pair 相关的位置非零
    向量化实现：在 ROI 内一次性构建权重，不再逐行 for 循环。
    """
    H, W = mask_ref.shape
    w_ref = np.zeros((H, W), dtype=np.float32)
    w_tgt = np.zeros((H, W), dtype=np.float32)

    x_min, y_min, x_max, y_max = bbox
    H_roi = y_max - y_min + 1
    W_roi = x_max - x_min + 1

    # 截取 ROI
    overlap_roi = overlap[y_min:y_max+1, x_min:x_max+1]  # (H_roi, W_roi)

    # 如果这个 ROI 实际上没有 overlap，直接返回全零
    if not overlap_roi.any():
        return w_ref, w_tgt

    # seam_x_roi 是 ROI 内局部 x 坐标（长度 H_roi）
    seam_x_roi = np.asarray(seam_x_roi, dtype=np.int32).reshape(H_roi, 1)  # (H_roi,1)

    # 这里所有计算都在 ROI 局部坐标 [0, W_roi-1] 上进行
    # 每行各自的 left/right
    left_roi  = np.clip(seam_x_roi - band_width, 0, W_roi - 1)  # (H_roi,1)
    right_roi = np.clip(seam_x_roi + band_width, 0, W_roi - 1)  # (H_roi,1)

    # 构造每行的 x 坐标网格
    xs = np.arange(W_roi, dtype=np.int32)[None, :]   # (1, W_roi)
    xs = np.repeat(xs, H_roi, axis=0)                # (H_roi, W_roi)

    # 三个区域的 mask（在 ROI 内）
    # 左侧：x < left
    mask_left  = xs < left_roi
    # 右侧：x > right
    mask_right = xs > right_roi
    # 中间带：既不在左也不在右
    mask_band  = ~(mask_left | mask_right)

    # 初始化 ROI 内权重
    w_ref_roi = np.zeros((H_roi, W_roi), dtype=np.float32)
    w_tgt_roi = np.zeros((H_roi, W_roi), dtype=np.float32)

    # 左侧：完全 ref
    w_ref_roi[mask_left]  = 1.0
    w_tgt_roi[mask_left]  = 0.0

    # 右侧：完全 tgt
    w_ref_roi[mask_right] = 0.0
    w_tgt_roi[mask_right] = 1.0

    # 中间带：线性过渡
    # t = (x - left) / max(1, right-left)
    denom = (right_roi - left_roi).astype(np.float32)  # (H_roi,1)
    denom[denom < 1.0] = 1.0

    t_full = (xs.astype(np.float32) - left_roi.astype(np.float32)) / denom  # (H_roi,W_roi)
    t_full = np.clip(t_full, 0.0, 1.0)

    w_tgt_roi[mask_band] = t_full[mask_band]
    w_ref_roi[mask_band] = 1.0 - w_tgt_roi[mask_band]

    # 只在 overlap 区有效，非 overlap 直接清零
    w_ref_roi[~overlap_roi] = 0.0
    w_tgt_roi[~overlap_roi] = 0.0

    # 写回全局画布
    w_ref[y_min:y_max+1, x_min:x_max+1] = w_ref_roi
    w_tgt[y_min:y_max+1, x_min:x_max+1] = w_tgt_roi

    return w_ref, w_tgt

def render_pano_seam(
    world_warp_list, world_masks,
    ref_id=0, band_width=16,
    w_c=1.0, w_g=2.0,
    forbid_masks=None,
    w_f=10.0,            # 掩膜惩罚系数
):
    """
    N 路 seam 融合（阶段 A：Ec + Eg）
    输入:
      world_warp_list: List[HxWx3]
      world_masks    : List[HxW]
      ref_id         : 参考视图索引
    输出:
      pano: HxWx3 uint8
    """
    N = len(world_warp_list)
    H, W = world_masks[0].shape
    weights = [np.zeros((H, W), dtype=np.float32) for _ in range(N)]

    # 把 mask 先堆成一个 (N,H,W) 方便后面做向量化
    masks = np.stack([(m > 0).astype(np.uint8) for m in world_masks], axis=0)  # (N,H,W)

    img_ref  = world_warp_list[ref_id]
    mask_ref = world_masks[ref_id]

    # 逐路跟 ref 做 seam（保持原逻辑）
    for i in range(N):
        if i == ref_id:
            continue
        img_tgt  = world_warp_list[i]
        mask_tgt = world_masks[i]

        overlap, bbox = compute_overlap_and_bbox(mask_ref, mask_tgt)
        if bbox is None:
            continue  # 与 ref 无重叠，交给后面“只属于某视图”的逻辑

        forbid_ref = None
        forbid_tgt = None
        if forbid_masks is not None:
            forbid_ref = forbid_masks[ref_id]
            forbid_tgt = forbid_masks[i]

        E_roi = compute_energy_Ec_Eg(img_ref, img_tgt, overlap, bbox,w_c=w_c, w_g=w_g,forbid_ref=forbid_ref,forbid_tgt=forbid_tgt,w_f=w_f,)
        seam_x_roi = find_vertical_seam_dp(E_roi)
        w_ref_pair, w_tgt_pair = build_pair_weights_from_seam(
            mask_ref, mask_tgt, overlap, bbox, seam_x_roi,
            band_width=band_width
        )

        weights[ref_id] += w_ref_pair
        weights[i]      += w_tgt_pair

    # ==================== 向量化处理 unset 像素 ====================

    # 当前已有的权重和
    weights_sum = np.zeros((H, W), dtype=np.float32)
    for k in range(N):
        weights_sum += weights[k]

    # 至少被某路覆盖的像素
    present_count = masks.sum(axis=0)          # (H,W)，有几路覆盖
    any_mask = present_count > 0
    unset   = (weights_sum == 0) & any_mask    # 还没被 seam 分配权重、但有视图覆盖

    # 情况 1：只被一条视图覆盖 → 那条视图权重 = 1
    single = unset & (present_count == 1)      # (H,W)
    if np.any(single):
        owner = masks.argmax(axis=0)           # (H,W)，只有一条是1，所以 argmax 就是那一路
        for k in range(N):
            wk = weights[k]
            mask_k_single = single & (owner == k)
            if np.any(mask_k_single):
                wk[mask_k_single] = 1.0
                weights[k] = wk

    # 情况 2：多路覆盖，但尚未被 seam 分配的像素
    multi = unset & (present_count > 1)
    if np.any(multi):
        if 0 <= ref_id < N:
            # 2a. 优先给 ref：multi 中，且 ref 覆盖到的那些像素 → ref=1
            ref_cover = multi & (masks[ref_id] > 0)
            if np.any(ref_cover):
                weights[ref_id][ref_cover] = 1.0

            # 2b. 剩余 multi 且没有 ref 的像素 → 在所有覆盖视图之间均分
            rest = multi & ~ref_cover
            if np.any(rest):
                for k in range(N):
                    mk = masks[k] > 0
                    mask_k = rest & mk
                    if np.any(mask_k):
                        wk = weights[k]
                        wk[mask_k] = 1.0 / present_count[mask_k].astype(np.float32)
                        weights[k] = wk
        else:
            # 没有 ref_id 的情况：multi 里全部均分
            for k in range(N):
                mk = masks[k] > 0
                mask_k = multi & mk
                if np.any(mask_k):
                    wk = weights[k]
                    wk[mask_k] = 1.0 / present_count[mask_k].astype(np.float32)
                    weights[k] = wk

    # ==================== 归一化 + 按 N 路权重融合 ====================
    
    weights_sum = np.zeros((H, W), dtype=np.float32)
    for k in range(N):
        weights_sum += weights[k]
    valid = weights_sum > 1e-6
    for k in range(N):
        wk = weights[k]
        wk[valid] /= weights_sum[valid]
        weights[k] = wk

    pano_f = np.zeros_like(world_warp_list[0], dtype=np.float32)
    for k in range(N):
        pano_f += world_warp_list[k].astype(np.float32) * weights[k][..., None]

    pano = np.clip(pano_f, 0, 255).astype(np.uint8)
    return pano

# ===================================================================
# 对所有重叠对做 seam，避免未参与 seam 的重叠被均分
# ===================================================================
def render_pano_seam(
    world_warp_list, world_masks,
    ref_id=0, band_width=16,
    w_c=1.0, w_g=2.0,
    forbid_masks=None,
    w_f=10.0,
):
    N = len(world_warp_list)
    H, W = world_masks[0].shape
    weights = [np.zeros((H, W), dtype=np.float32) for _ in range(N)]

    masks = np.stack([(m > 0).astype(np.uint8) for m in world_masks], axis=0)  # (N,H,W)

    # 对所有实际有重叠的视图对做 seam
    for i in range(N):
        for j in range(i + 1, N):
            overlap, bbox = compute_overlap_and_bbox(world_masks[i], world_masks[j])
            if bbox is None:
                continue

            forbid_i = None
            forbid_j = None
            if forbid_masks is not None:
                forbid_i = forbid_masks[i]
                forbid_j = forbid_masks[j]

            E_roi = compute_energy_Ec_Eg(
                world_warp_list[i], world_warp_list[j],
                overlap, bbox,
                w_c=w_c, w_g=w_g,
                forbid_ref=forbid_i, forbid_tgt=forbid_j,
                w_f=w_f,
            )
            seam_x_roi = find_vertical_seam_dp(E_roi)
            w_i_pair, w_j_pair = build_pair_weights_from_seam(
                world_masks[i], world_masks[j],
                overlap, bbox, seam_x_roi,
                band_width=band_width
            )
            weights[i] += w_i_pair
            weights[j] += w_j_pair

    # unset 像素处理，沿用原逻辑
    weights_sum = np.zeros((H, W), dtype=np.float32)
    for k in range(N):
        weights_sum += weights[k]

    present_count = masks.sum(axis=0)
    any_mask = present_count > 0
    unset = (weights_sum == 0) & any_mask

    single = unset & (present_count == 1)
    if np.any(single):
        owner = masks.argmax(axis=0)
        for k in range(N):
            wk = weights[k]
            mask_k_single = single & (owner == k)
            if np.any(mask_k_single):
                wk[mask_k_single] = 1.0
                weights[k] = wk

    multi = unset & (present_count > 1)
    if np.any(multi):
        if 0 <= ref_id < N:
            ref_cover = multi & (masks[ref_id] > 0)
            if np.any(ref_cover):
                weights[ref_id][ref_cover] = 1.0
            rest = multi & ~ref_cover
            if np.any(rest):
                for k in range(N):
                    mk = masks[k] > 0
                    mask_k = rest & mk
                    if np.any(mask_k):
                        wk = weights[k]
                        wk[mask_k] = 1.0 / present_count[mask_k].astype(np.float32)
                        weights[k] = wk
        else:
            for k in range(N):
                mk = masks[k] > 0
                mask_k = multi & mk
                if np.any(mask_k):
                    wk = weights[k]
                    wk[mask_k] = 1.0 / present_count[mask_k].astype(np.float32)
                    weights[k] = wk

    # 归一化 + 融合
    weights_sum = np.zeros((H, W), dtype=np.float32)
    for k in range(N):
        weights_sum += weights[k]
    valid = weights_sum > 1e-6
    for k in range(N):
        wk = weights[k]
        wk[valid] /= weights_sum[valid]
        weights[k] = wk

    pano_f = np.zeros_like(world_warp_list[0], dtype=np.float32)
    for k in range(N):
        pano_f += world_warp_list[k].astype(np.float32) * weights[k][..., None]

    pano = np.clip(pano_f, 0, 255).astype(np.uint8)
    return pano
