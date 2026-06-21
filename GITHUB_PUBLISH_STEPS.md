# GitHub 发布说明

建议在你的本地 Chrome 中新建仓库 `ECMamba`，然后在这个文件夹里执行下面的命令完成首次发布。

## 1. 在 GitHub 网站创建空仓库

- 仓库名：`ECMamba`
- 可见性：按你的需要选择 `Public` 或 `Private`
- 不要勾选自动添加 `README`、`.gitignore` 或 license，这样可以直接推送本地内容

## 2. 本地首次提交

在当前文件夹运行：

```bash
git branch -M main
git add README.md RESOURCE_ESTIMATE.md requirements.txt esm2_3b_msa_mamba_only_gated_split100_train_eval.py model_comparison_full.png assets/ ECMamba_Bioinformatics_Application_Note_Revised.docx .gitignore
git commit -m "Initial ECMamba release"
git remote add origin https://github.com/duanHY-26/ECMamba.git
git push -u origin main
```

## 3. 首次发布后建议检查

- `README.md` 是否正常渲染
- `ECMamba_Bioinformatics_Application_Note_Revised.docx` 是否已上传
- `assets/ecmamba_gate_equation.png` 是否可见
- 仓库首页 description 是否补充为一句方法简介

## 4. 建议后续补充

- 补充 `LICENSE`
- 补充 `CITATION.cff`
- 在 `README.md` 中加入数据获取方式和运行示例
- 如果后续补充真实测得的显存与时间，请同步更新 `RESOURCE_ESTIMATE.md`
