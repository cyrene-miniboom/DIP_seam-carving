import os
import cv2
import csv
import time
import argparse
import numpy as np


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ============================
# Baseline Seam Carving
# ============================
def energy_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.abs(dx) + np.abs(dy)


def find_vertical_seam(energy):
    h, w = energy.shape
    dp = energy.copy()
    backtrack = np.zeros((h, w), dtype=np.int32)

    for i in range(1, h):
        for j in range(w):
            left = max(j - 1, 0)
            right = min(j + 1, w - 1)
            prev = dp[i - 1, left:right + 1]
            idx = int(np.argmin(prev))
            backtrack[i, j] = left + idx
            dp[i, j] += prev[idx]

    seam = np.zeros(h, dtype=np.int32)
    seam[h - 1] = int(np.argmin(dp[h - 1]))

    for i in range(h - 2, -1, -1):
        seam[i] = backtrack[i + 1, seam[i + 1]]

    return seam  # (h,)


def insert_vertical_seam(img, seam):
    h, w, c = img.shape
    out = np.zeros((h, w + 1, c), dtype=img.dtype)

    for i in range(h):
        j = int(seam[i])
        out[i, :j, :] = img[i, :j, :]

        if j == 0:
            p = img[i, j, :]
        else:
            p = ((img[i, j - 1, :].astype(np.int32) + img[i, j, :].astype(np.int32)) // 2).astype(img.dtype)

        out[i, j, :] = p
        out[i, j + 1:, :] = img[i, j:, :]

    return out


def seam_insert_baseline(img, add_pixels):
    out = img.copy()
    mid_cols = []
    t0 = time.perf_counter()

    for _ in range(add_pixels):
        e = energy_map(out)
        seam = find_vertical_seam(e)
        mid_cols.append(int(seam[len(seam) // 2]))
        out = insert_vertical_seam(out, seam)

    dt = time.perf_counter() - t0
    return out, dt, mid_cols


# ============================
# Improved Seam Carving (yours)
# ROI + smooth + column penalty + multi-band
# ============================
def energy_map_with_roi(img, roi_list=None, roi_weight=60.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    energy = np.abs(dx) + np.abs(dy)

    if not roi_list:
        return energy

    h, w = gray.shape
    roi_mask = np.zeros((h, w), dtype=np.float32)
    for (x1, y1, x2, y2) in roi_list:
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))
        if x2 > x1 and y2 > y1:
            roi_mask[y1:y2, x1:x2] = 1.0

    max_e = float(np.max(energy)) if np.max(energy) > 0 else 1.0
    return energy + roi_weight * max_e * roi_mask


def smooth_seam(seam_pairs, width, window=5):
    h = len(seam_pairs)
    cols = np.array([c for (_, c) in seam_pairs], dtype=np.float32)

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
            prev = int(new_cols[i - 1])
            candidates = [prev]
            if prev - 1 >= 0:
                candidates.append(prev - 1)
            if prev + 1 < width:
                candidates.append(prev + 1)
            j = min(candidates, key=lambda x: abs(x - smooth_cols[i]))
        new_cols[i] = j

    return [(i, int(new_cols[i])) for i in range(h)]


def insert_column_penalty(col_penalty, j_seam, extra_bias):
    w = col_penalty.shape[0]
    new_penalty = np.zeros(w + 1, dtype=np.float32)
    new_penalty[:j_seam] = col_penalty[:j_seam]
    new_penalty[j_seam] = col_penalty[j_seam] + extra_bias
    new_penalty[j_seam + 1:] = col_penalty[j_seam:]
    return new_penalty


def choose_preferred_columns(col_cost, num_bands=12, min_gap=15):
    w = len(col_cost)
    chosen = []
    for _ in range(num_bands):
        mask = np.ones(w, dtype=bool)
        for c in chosen:
            left = max(0, c - min_gap)
            right = min(w, c + min_gap + 1)
            mask[left:right] = False

        cand = np.where(mask)[0]
        if cand.size == 0:
            break
        j = int(cand[np.argmin(col_cost[cand])])
        chosen.append(j)
    return chosen


def seam_insert_ours(
    img,
    add_pixels,
    roi_list=None,
    roi_weight=60.0,
    smooth_window=5,
    use_column_penalty=True,
    lambda_penalty=1.0,
    extra_bias=5.0,
    use_multi_band=True,
    num_bands=12,
    min_gap=15,
    band_bias_ratio=0.4,
):
    out = img.copy()
    column_penalty = np.zeros(out.shape[1], dtype=np.float32)
    mid_cols = []
    t0 = time.perf_counter()

    for _ in range(add_pixels):
        h, w = out.shape[:2]

        # sync penalty length
        if column_penalty.shape[0] != w:
            if column_penalty.shape[0] < w:
                column_penalty = np.pad(column_penalty, (0, w - column_penalty.shape[0]), mode="edge")
            else:
                column_penalty = column_penalty[:w]

        base_energy = energy_map_with_roi(out, roi_list=roi_list, roi_weight=roi_weight)
        energy = base_energy.copy()

        # multi-band bias
        if use_multi_band:
            col_cost = base_energy.sum(axis=0)
            preferred = choose_preferred_columns(col_cost, num_bands=num_bands, min_gap=min_gap)
            if preferred:
                preferred_mask = np.zeros(w, dtype=np.float32)
                preferred_mask[preferred] = 1.0
                max_e = float(np.max(base_energy)) if np.max(base_energy) > 0 else 1.0
                energy = energy - band_bias_ratio * max_e * preferred_mask[np.newaxis, :]

        # column penalty
        if use_column_penalty:
            energy = energy + lambda_penalty * column_penalty[np.newaxis, :]

        seam_cols = find_vertical_seam(energy)
        seam_pairs = [(i, int(seam_cols[i])) for i in range(h)]
        seam_pairs = smooth_seam(seam_pairs, width=w, window=smooth_window)

        # record mid column
        j_mid = int(seam_pairs[len(seam_pairs) // 2][1])
        mid_cols.append(j_mid)

        # update penalty then insert seam
        if use_column_penalty:
            cols = [c for (_, c) in seam_pairs]
            for j in set(cols):
                column_penalty[j] += 1.0
            column_penalty = insert_column_penalty(column_penalty, j_mid, extra_bias)

        out = insert_vertical_seam(out, np.array([c for (_, c) in seam_pairs], dtype=np.int32))

    dt = time.perf_counter() - t0
    return out, dt, mid_cols


# ============================
# Utilities
# ============================
def list_images(folder):
    files = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and name.lower().endswith(IMG_EXTS):
            files.append(p)
    files.sort()
    return files


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def pad_to_width(img, target_w):
    h, w = img.shape[:2]
    if w >= target_w:
        return img
    pad = target_w - w
    return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def mid_stats(mid_cols):
    if not mid_cols:
        return 0, 0
    span = int(max(mid_cols) - min(mid_cols))
    # max frequency
    vals, counts = np.unique(np.array(mid_cols), return_counts=True)
    maxfreq = int(np.max(counts))
    return span, maxfreq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str,
                    default=os.path.join("data", "OpenDataLab___BSDS500", "mini_bsds500"),
                    help="folder of your selected 25 images")
    ap.add_argument("--out_dir", type=str, default="results_bsds500", help="output folder")
    ap.add_argument("--add_pixels", type=int, default=80, help="increase width by this many pixels")
    ap.add_argument("--roi_lenna", action="store_true",
                    help="apply a fixed ROI for images whose filename contains 'lenna' (optional)")
    args = ap.parse_args()

    input_dir = args.input_dir
    out_dir = args.out_dir
    add_pixels = args.add_pixels

    imgs = list_images(input_dir)
    if not imgs:
        raise SystemExit(f"❌ No images found in: {input_dir}")

    out_resize = os.path.join(out_dir, "resize")
    out_base = os.path.join(out_dir, "baseline")
    out_ours = os.path.join(out_dir, "ours")
    out_cmp = os.path.join(out_dir, "compare")
    ensure_dir(out_resize); ensure_dir(out_base); ensure_dir(out_ours); ensure_dir(out_cmp)

    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "filename", "H", "W",
            "add_pixels",
            "resize_time_s", "baseline_time_s", "ours_time_s",
            "baseline_mid_span", "ours_mid_span",
            "baseline_mid_maxfreq", "ours_mid_maxfreq",
        ])

        for idx, p in enumerate(imgs, 1):
            name = os.path.basename(p)
            img = cv2.imread(p)
            if img is None:
                print(f"[{idx}/{len(imgs)}] Skip unreadable: {name}")
                continue

            h, w0 = img.shape[:2]
            target_w = w0 + add_pixels

            # 1) resize
            t0 = time.perf_counter()
            resize_img = cv2.resize(img, (target_w, h), interpolation=cv2.INTER_LINEAR)
            t_resize = time.perf_counter() - t0

            # 2) baseline seam
            base_img, t_base, base_mids = seam_insert_baseline(img, add_pixels)

            # 3) ours seam (ROI optional only for lenna)
            roi_list = None
            if args.roi_lenna and ("lenna" in name.lower()):
                # 你之前用的 ROI（如果你想让 lenna 单独启用 ROI）
                roi_list = [(46, 0, 226, h)]

            ours_img, t_ours, ours_mids = seam_insert_ours(img, add_pixels, roi_list=roi_list)

            # save images
            cv2.imwrite(os.path.join(out_resize, name), resize_img)
            cv2.imwrite(os.path.join(out_base, name), base_img)
            cv2.imwrite(os.path.join(out_ours, name), ours_img)

            # compare montage: [orig(padded), resize, baseline, ours]
            orig_pad = pad_to_width(img, target_w)
            cmp = cv2.hconcat([orig_pad, resize_img, base_img, ours_img])
            cv2.imwrite(os.path.join(out_cmp, name), cmp)

            b_span, b_maxf = mid_stats(base_mids)
            o_span, o_maxf = mid_stats(ours_mids)

            wcsv.writerow([
                name, h, w0,
                add_pixels,
                round(t_resize, 4), round(t_base, 4), round(t_ours, 4),
                b_span, o_span,
                b_maxf, o_maxf,
            ])

            print(f"[{idx}/{len(imgs)}] {name} done | resize {t_resize:.2f}s | base {t_base:.2f}s | ours {t_ours:.2f}s")

    print("✅ All done.")
    print(f"  - Results folder: {out_dir}")
    print(f"  - Summary CSV:    {csv_path}")
    print("  - Compare images: results_bsds500/compare/ (best for paper figures)")


if __name__ == "__main__":
    main()
