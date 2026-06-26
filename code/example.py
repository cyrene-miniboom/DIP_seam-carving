import cv2
import numpy as np

# =========================
# 1. 原始 Seam Carving（baseline）
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
# 2. 改进算法 + ROI 保护
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


def seam_carving_roi(img, add_pixels, roi):
    out = img.copy()
    for _ in range(add_pixels):
        energy = energy_map_with_roi(out, roi)
        seam = find_vertical_seam(energy)
        out = insert_vertical_seam(out, seam)
    return out


# =========================
# 3. 主对比流程
# =========================
if __name__ == "__main__":
    img = cv2.imread(r"D:\anaconda\envs\xiangmu\pythonProject2\data\OpenDataLab___BSDS500\mini_bsds500\sample_45.jpg"
)
    if img is None:
        raise SystemExit("❌ 找不到 sample45.jpg")

    h, w = img.shape[:2]
    add_pixels = 80
    target_w = w + add_pixels

    # 原图 padding
    orig_pad = cv2.copyMakeBorder(
        img, 0, 0, 0, add_pixels,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # 普通 resize
    resize_img = cv2.resize(img, (target_w, h))

    # 原始 Seam Carving
    baseline = seam_carving_baseline(img, add_pixels)

    # 改进算法 + ROI（人像区域）
    roi = (75, 26, 248, 480)  # 包住人像
    ours_roi = seam_carving_roi(img, add_pixels, roi)

    # 拼接对比
    compare = cv2.hconcat([
        orig_pad,
        resize_img,
        baseline,
        ours_roi
    ])

    cv2.imwrite("example1.png", compare)
    print("✅ 已生成 example1.png")
