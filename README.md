# 基于 Seam Carving 的内容保持型图像拉伸算法

这是根据论文《基于 Seam Carving 的内容保持型图像拉伸算法》制作的 GitHub Pages 项目介绍页。

## 本版本的特点

- `index.html` 与 `style.css` 分离；
- 页面中的实验图片、折线图、柱状图均直接从论文 Word 文件中提取；
- 页面包含论文表 4-1 的 25 张图片完整统计数据；
- 页面包含论文表 4-2 的平均结果；
- 不再使用图片占位图；
- 响应式布局，可在电脑和手机浏览器中使用。

## 文件结构

```text
seam-carving-github-page/
├── index.html
├── style.css
├── README.md
├── IMAGE_GUIDE.md
└── images/
    ├── lenna-original-result.png
    ├── lenna-improved-result.png
    ├── sample-36-comparison.png
    ├── sample-105-comparison.png
    ├── sample-stairs-comparison.png
    ├── sample-animals-comparison.png
    ├── chart-mid-maxfreq-line.png
    ├── chart-mid-span-line.png
    ├── chart-mid-maxfreq-average.png
    ├── chart-mid-span-average.png
    ├── roi-without-protection.png
    ├── roi-with-protection.png
    └── ablation-sample45.png
```

## 发布到 GitHub Pages

1. 新建一个 GitHub 仓库；
2. 将本文件夹内的全部文件上传到仓库根目录；
3. 在 `index.html` 中将所有 `YOUR_NAME/YOUR_REPO` 替换为实际仓库路径；
4. 进入仓库 `Settings → Pages`；
5. 在 `Build and deployment` 中选择 `Deploy from a branch`；
6. 选择 `main` 分支和 `/root` 目录；
7. 保存并等待 GitHub Pages 发布。
