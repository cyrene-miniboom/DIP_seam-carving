# Improved Seam Carving GitHub Project Page

这是根据论文《基于 Seam Carving 的内容保持型图像拉伸算法》整理的 GitHub 项目介绍网页。

## 文件结构

```text
seam-carving-github-page/
├── index.html
├── style.css
├── IMAGE_GUIDE.md
└── images/
    ├── hero-comparison.svg
    ├── algorithm-workflow.svg
    ├── chart-mid-maxfreq.svg
    ├── chart-mid-span.svg
    ├── qualitative-comparison.svg
    ├── roi-comparison.svg
    └── ablation-study.svg
```

## 使用方法

1. 将整个文件夹放入 GitHub 仓库。
2. 用你的真实仓库地址替换 `index.html` 中的：
   - `https://github.com/YOUR_NAME/YOUR_REPO`
3. 按照 `IMAGE_GUIDE.md` 的说明，用论文图片或程序输出替换 `images/` 中的占位图。
4. 保持文件名不变，网页会自动显示替换后的图片。
5. 在 GitHub 仓库的 `Settings → Pages` 中启用 GitHub Pages。

## 页面内容

- 项目背景与意义
- 原始 Seam Carving 的问题
- ROI、路径平滑、列惩罚、多带扩展四项改进
- mini-BSDS500 实验设置
- mid_maxfreq / mid_span 定量结果
- 定性对比、ROI 对比与消融实验
- 总结、未来工作与项目成员
