import cv2
import numpy as np

# =========================
# 1. 原始 Seam Carving
# =========================
def energy_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return np.abs(dx) + np.abs(dy)


def find_vertical_seam(energy):
    h, w = energy.shape
    dp = energy.copy()
    back = np.zeros_like(dp, dtype=np.int32)

    for i in range(1, h):
        for j in range(w):
            l = max(j - 1, 0)
            r = min(j + 1, w - 1)
            idx = np.argmin(dp[i - 1, l:r + 1])
            back[i, j] = l + idx
            dp[i, j] += dp[i - 1, l + idx]

    seam = []
    j = np.argmin(dp[-1])
    for i in reversed(range(h)):
        seam.append((i, j))
        j = back[i, j]
    seam.reverse()
    return seam


def insert_vertical_seam(img, seam):
    h, w, c = img.shape
    out = np.zeros((h, w + 1, c), dtype=img.dtype)
    for i, (_, j) in enumerate(seam):
        out[i, :j] = img[i, :j]
        out[i, j] = img[i, j]
        out[i, j + 1:] = img[i, j:]
    return out


def seam_carving_baseline(img, add_pixels):
    out = img.copy()
    for _ in range(add_pixels):
        energy = energy_map(out)
        seam = find_vertical_seam(energy)
        out = insert_vertical_seam(out, seam)
    return out


# =========================
# 2. ROI 能量保护
# =========================
def energy_map_with_roi(img, roi, roi_weight=60):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    energy = np.abs(dx) + np.abs(dy)

    mask = np.zeros_like(gray, dtype=np.float32)
    x1, y1, x2, y2 = roi
    mask[y1:y2, x1:x2] = 1.0
    energy += roi_weight * np.max(energy) * mask
    return energy


def smooth_seam(seam, width, window=5):
    cols = np.array([j for (_, j) in seam], dtype=np.float32)
    if window > 1:
        if window % 2 == 0:
            window += 1
        k = np.ones(window) / window
        cols = np.convolve(np.pad(cols, window//2, mode="edge"), k, mode="valid")

    new_cols = np.zeros_like(cols, dtype=np.int32)
    for i in range(len(cols)):
        if i == 0:
            new_cols[i] = int(cols[i])
        else:
            prev = new_cols[i - 1]
            cand = [prev]
            if prev > 0: cand.append(prev - 1)
            if prev < width - 1: cand.append(prev + 1)
            new_cols[i] = min(cand, key=lambda x: abs(x - cols[i]))

    return [(i, int(new_cols[i])) for i in range(len(new_cols))]


# =========================
# 3. 带惩罚的 Seam Carving（用于消融）
# =========================
def seam_carving_roi(
    img,
    add_pixels,
    roi,
    use_smooth=False,
    use_penalty=False,
    lambda_penalty=1.0
):
    out = img.copy()
    col_penalty = np.zeros(out.shape[1], dtype=np.float32)

    for _ in range(add_pixels):
        h, w = out.shape[:2]

        # ===== 修复点：同步列惩罚长度 =====
        if use_penalty:
            if col_penalty.shape[0] < w:
                col_penalty = np.pad(col_penalty, (0, w - col_penalty.shape[0]), mode="edge")
            elif col_penalty.shape[0] > w:
                col_penalty = col_penalty[:w]

        energy = energy_map_with_roi(out, roi)

        if use_penalty:
            energy = energy + lambda_penalty * col_penalty[np.newaxis, :]

        seam = find_vertical_seam(energy)

        if use_smooth:
            seam = smooth_seam(seam, w)

        if use_penalty:
            cols = [j for (_, j) in seam]
            for j in set(cols):
                col_penalty[j] += 1.0

        out = insert_vertical_seam(out, seam)

    return out


# =========================
# 4. 消融实验主流程
# =========================
if __name__ == "__main__":
    img = cv2.imread(
        r"D:\anaconda\envs\xiangmu\pythonProject2\data\OpenDataLab___BSDS500\mini_bsds500\sample_45.jpg"
    )
    if img is None:
        raise SystemExit("❌ 找不到 sample_45.jpg")

    h, w = img.shape[:2]
    add_pixels = 80
    target_w = w + add_pixels

    roi = (75, 26, 248, 480)  # 人像 ROI

    # 原图 padding
    orig = cv2.copyMakeBorder(img, 0, 0, 0, add_pixels, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    resize = cv2.resize(img, (target_w, h))
    baseline = seam_carving_baseline(img, add_pixels)

    roi_only = seam_carving_roi(img, add_pixels, roi)
    roi_smooth = seam_carving_roi(img, add_pixels, roi, use_smooth=True)
    roi_smooth_penalty = seam_carving_roi(
        img, add_pixels, roi,
        use_smooth=True,
        use_penalty=True
    )

    compare = cv2.hconcat([
        orig,
        resize,
        baseline,
        roi_only,
        roi_smooth,
        roi_smooth_penalty
    ])

    cv2.imwrite("example_ablation.png", compare)
    print("✅ 已生成 example_ablation.png")
