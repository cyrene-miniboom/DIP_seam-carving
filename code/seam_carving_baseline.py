import cv2
import numpy as np


# ============================
# 1) 能量图（Sobel）
# ============================
def energy_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(dx) + np.abs(dy)
    return energy


# ============================
# 2) 动态规划：找最小能量竖直 seam
# seam 是一条从上到下的路径，每行一个列坐标
# ============================
def find_vertical_seam(energy):
    h, w = energy.shape
    dp = energy.copy()
    backtrack = np.zeros((h, w), dtype=np.int32)

    for i in range(1, h):
        for j in range(w):
            left = max(j - 1, 0)
            right = min(j + 1, w - 1)
            prev = dp[i - 1, left:right + 1]
            idx = np.argmin(prev)
            backtrack[i, j] = left + idx
            dp[i, j] += prev[idx]

    seam = np.zeros(h, dtype=np.int32)
    seam[h - 1] = np.argmin(dp[h - 1])

    for i in range(h - 2, -1, -1):
        seam[i] = backtrack[i + 1, seam[i + 1]]

    return seam  # shape=(h,)


# ============================
# 3) 插入竖直 seam（图像宽度 +1）
# 每一行在 seam[j] 位置插入一个像素（用邻域平均）
# ============================
def insert_vertical_seam(img, seam):
    h, w, c = img.shape
    out = np.zeros((h, w + 1, c), dtype=img.dtype)

    for i in range(h):
        j = int(seam[i])

        # 左边直接拷贝
        out[i, :j, :] = img[i, :j, :]

        # 插入像素：取 seam 位置与其左侧的平均（j=0 时直接取自身）
        if j == 0:
            p = img[i, j, :]
        else:
            p = ((img[i, j - 1, :].astype(np.int32) + img[i, j, :].astype(np.int32)) // 2).astype(img.dtype)

        out[i, j, :] = p

        # 右边拷贝（原来的 j..w-1 往右挪一格）
        out[i, j + 1:, :] = img[i, j:, :]

    return out


# ============================
# 4) 主程序：横向拉宽 add_pixels 像素
# ============================
if __name__ == "__main__":
    input_path = "lenna.jpg"          # 改成你的图片名
    add_pixels = 80                   # 横向拉宽多少像素
    out_path = "baseline_seam_add.png"

    img = cv2.imread(input_path)
    if img is None:
        print("❌ 没找到图像，请检查文件名/路径：", input_path)
        raise SystemExit(1)

    out = img.copy()

    for k in range(add_pixels):
        energy = energy_map(out)
        seam = find_vertical_seam(energy)
        out = insert_vertical_seam(out, seam)

        if (k + 1) % 10 == 0:
            print(f"已插入 {k + 1} 条 seam...")

    cv2.imwrite(out_path, out)
    print("✅ 完成：", out_path)
