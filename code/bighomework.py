import cv2
import numpy as np


# ============================
# 1. 计算能量图（Sobel + ROI 高权重）
# ============================
def energy_map_with_roi(img, roi_list=None, roi_weight=20.0):
    """
    :param img: BGR 图像
    :param roi_list: [(x1, y1, x2, y2), ...] 要保护的矩形区域（像素坐标）
    :param roi_weight: ROI 区域能量放大倍数（越大越难被 seam 穿过）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel 梯度作为基础能量
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    energy = np.abs(dx) + np.abs(dy)

    h, w = gray.shape
    roi_mask = np.zeros((h, w), dtype=np.float32)

    if roi_list is not None:
        for (x1, y1, x2, y2) in roi_list:
            # 防止越界
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                roi_mask[y1:y2, x1:x2] = 1.0

    if roi_mask.max() > 0:
        max_e = np.max(energy)
        # 在 ROI 区域大幅增加能量
        energy = energy + roi_weight * max_e * roi_mask

    return energy


# ============================
# 2. 动态规划寻找竖直 seam
# ============================
def find_vertical_seam(energy):
    h, w = energy.shape
    dp = energy.copy()
    backtrack = np.zeros_like(dp, dtype=np.int32)

    for i in range(1, h):
        for j in range(w):
            left = max(j - 1, 0)
            right = min(j + 1, w - 1)
            prev_row = dp[i - 1, left:right + 1]
            idx = np.argmin(prev_row)
            backtrack[i, j] = left + idx
            dp[i, j] += prev_row[idx]

    seam = []
    j = np.argmin(dp[-1])
    for i in reversed(range(h)):
        seam.append((i, j))
        j = backtrack[i, j]

    seam.reverse()
    return seam


# ============================
# 3. seam 平滑 + 连续约束
# ============================
def smooth_seam(seam, width, window=5):
    h = len(seam)
    cols = np.array([c for (_, c) in seam], dtype=np.float32)

    if window > 1:
        if window % 2 == 0:
            window += 1
        k = np.ones(window, dtype=np.float32)
        k /= k.sum()
        pad = window // 2
        cols_padded = np.pad(cols, (pad, pad), mode="edge")
        smooth_cols = np.convolve(cols_padded, k, mode="valid")
    else:
        smooth_cols = cols.copy()

    new_cols = np.zeros_like(cols, dtype=np.int32)
    for i in range(h):
        if i == 0:
            j = int(round(smooth_cols[i]))
            j = max(0, min(width - 1, j))
        else:
            candidates = [new_cols[i - 1]]
            if new_cols[i - 1] - 1 >= 0:
                candidates.append(new_cols[i - 1] - 1)
            if new_cols[i - 1] + 1 < width:
                candidates.append(new_cols[i - 1] + 1)

            best = min(candidates, key=lambda x: abs(x - smooth_cols[i]))
            j = best

        new_cols[i] = j

    new_seam = [(i, int(new_cols[i])) for i in range(h)]
    return new_seam


# ============================
# 4. 插入一条竖直 seam
# ============================
def insert_vertical_seam(img, seam):
    h, w, c = img.shape
    new_img = np.zeros((h, w + 1, c), dtype=np.uint8)

    for i in range(h):
        j_seam = seam[i][1]
        for ch in range(c):
            new_img[i, :j_seam, ch] = img[i, :j_seam, ch]

            if j_seam == 0:
                p = img[i, j_seam, ch]
            else:
                p = (int(img[i, j_seam - 1, ch]) + int(img[i, j_seam, ch])) // 2
            new_img[i, j_seam, ch] = p

            new_img[i, j_seam + 1:, ch] = img[i, j_seam:, ch]

    return new_img


# ============================
# 5. 列惩罚插入（惩罚机制相关）
# ============================
def insert_column_penalty(col_penalty, j_seam, extra_bias):
    """
    col_penalty: shape = (w,)
    在第 j_seam 列插入一个新列，并给这个新列加上额外偏置惩罚
    """
    w = col_penalty.shape[0]
    new_penalty = np.zeros(w + 1, dtype=np.float32)
    new_penalty[:j_seam] = col_penalty[:j_seam]
    new_penalty[j_seam] = col_penalty[j_seam] + extra_bias
    new_penalty[j_seam + 1:] = col_penalty[j_seam:]
    return new_penalty


# ============================
# 6. 选择多条“扩展带”列索引（新的机制）
# ============================
def choose_preferred_columns(col_cost, num_bands=3, min_gap=10):
    """
    在列代价 col_cost 中选择若干个“扩展带”列：
    - num_bands: 选多少条带
    - min_gap: 带与带之间至少隔多少列
    """
    w = len(col_cost)
    chosen = []
    used = np.zeros(w, dtype=bool)

    for _ in range(num_bands):
        mask = np.ones(w, dtype=bool)
        for c in chosen:
            left = max(0, c - min_gap)
            right = min(w, c + min_gap + 1)
            mask[left:right] = False

        cand_idx = np.where(mask)[0]
        if cand_idx.size == 0:
            break

        j = cand_idx[np.argmin(col_cost[cand_idx])]
        chosen.append(int(j))

    return chosen


# ============================
# 7. 主流程
# ============================
if __name__ == "__main__":
    # 1) 改成你的图片名
    img = cv2.imread("lenna.jpg")
    if img is None:
        print("❌ 没找到图像，请检查文件名")
        exit()

    h0, w0 = img.shape[:2]

    # 2) 手动指定要保护的区域（像素坐标）
    roi_list = [
        (46, 0, 226, 255),   # 你现在用的那一块，可以按需要微调
    ]

    add_pixels = 80          # 横向拉宽多少像素
    roi_weight = 60        # ROI 能量放大倍数（不够就调大）
    smooth_window = 5        # seam 平滑窗口

    # ==== 惩罚机制参数（防止 seam 总走同一列）====
    use_column_penalty = True
    lambda_penalty = 1.0        # 惩罚权重（0.5~3.0）
    extra_bias = 5.0            # 新插入列的额外惩罚

    # ==== 新机制：多带扩展偏置 ====
    use_multi_band = True       # 开关：是否启用多带扩展
    num_bands = 12              # 每次迭代偏置 3 条扩展带
    min_gap = 15                # 带与带之间至少相隔 10 列
    band_bias_ratio = 0.4       # 对带列减去的能量系数（0.2~0.8 之间试）

    # 普通拉伸对比
    normal = cv2.resize(img, (w0 + add_pixels, h0), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("output_normal_stretch.png", normal)

    out = img.copy()
    column_penalty = np.zeros(out.shape[1], dtype=np.float32)

    for step in range(add_pixels):
        h, w, _ = out.shape

        # 宽度变化时，同步列惩罚长度
        if column_penalty.shape[0] != w:
            if column_penalty.shape[0] < w:
                column_penalty = np.pad(
                    column_penalty, (0, w - column_penalty.shape[0]), mode="edge"
                )
            else:
                column_penalty = column_penalty[:w]

        # 先算带 ROI 保护的基础能量
        base_energy = energy_map_with_roi(out, roi_list=roi_list, roi_weight=roi_weight)
        energy = base_energy.copy()

        # ===== 新机制：多带扩展偏置 =====
        if use_multi_band:
            # 列代价：一列上的能量和
            col_cost = base_energy.sum(axis=0)
            # 选择若干条低能量、间隔开的“扩展带”
            preferred_cols = choose_preferred_columns(
                col_cost, num_bands=num_bands, min_gap=min_gap
            )
            if len(preferred_cols) > 0:
                preferred_mask = np.zeros(w, dtype=np.float32)
                preferred_mask[preferred_cols] = 1.0
                max_e = np.max(base_energy)
                # 对扩展带所在列减去一部分能量 → seam 更爱走这些列
                energy = energy - band_bias_ratio * max_e * preferred_mask[np.newaxis, :]

        # ===== 列惩罚（避免总走同一条列） =====
        if use_column_penalty:
            energy = energy + lambda_penalty * column_penalty[np.newaxis, :]

        # 找 seam + 平滑
        seam = find_vertical_seam(energy)
        seam = smooth_seam(seam, width=w, window=smooth_window)

        # 更新列惩罚
        if use_column_penalty:
            cols = [c for (_, c) in seam]
            j_mid = cols[len(cols) // 2]
            for j in set(cols):
                column_penalty[j] += 1.0
            column_penalty = insert_column_penalty(column_penalty, j_mid, extra_bias)

        # 插入 seam
        out = insert_vertical_seam(out, seam)

        if (step + 1) % 10 == 0:
            print(f"已插入 {step + 1} 条 seam...")

    cv2.imwrite("finnal.png", out)
    print("✅ 完成：")
    print("  - output_normal_stretch.png                           普通横向拉伸")
    print("  - finnal.png       ROI 保护 + 平滑 + 列惩罚 + 多带扩展")