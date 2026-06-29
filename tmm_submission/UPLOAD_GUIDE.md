# IEEE TMM 投稿包

**Manuscript:** *An End-to-End Multimodal System for Subtitle Recognition and Chinese–Japanese Translation with Speech Synthesis in Short Dramas*
**Submission system:** ScholarOne Manuscripts — <https://mc.manuscriptcentral.com/tmm-ieee>
**作者:** Qiming Bao（唯一第一作者）, Jing An（**通讯作者**）, Rui-Yang Ju, Haofei Chang, Jinhua Su, Yanbing Bai, Xin Qu, Michael Witbrock

> ⚠️ **编译引擎必须用 XeLaTeX**（正文与补充材料含中日文 `xeCJK`，pdfLaTeX 会报错）。Overleaf: Menu → Compiler → XeLaTeX。
> ⚠️ TMM 评审是 **single-blind**（reviewer 看不到审稿意见，但作者署名公开在稿件上）。`main.tex` 已写入完整作者名/单位/邮箱（IEEEtran `\thanks{}`），自引以正常署名呈现。
> ⚠️ 正文本地实测 **10 页**（含参考文献，符合 TMM 初投上限）；作者简介用 `\iffalse` 关闭（不计页数，camera-ready 再开）。

---

## ScholarOne "Upload Manuscript" 槽位映射

ScholarOne 的 Upload 页面分 **Required Files** 和 **Optional Files** 两块，按槽位上传对应文件：

### 🔴 Required Files（必填）

| ScholarOne 槽位 | 上传文件 | 说明 |
|---|---|---|
| **Main Manuscript** | `main_manuscript.zip` | LaTeX 打包（根目录含 `main.tex` + `refs.bib` + `fig_1_v2.png`）。**不含**任何补充材料。系统会自动编译出审稿 PDF |
| **Conflict of Interest** | 勾选 ☑ "None of the authors have a conflict of interest to disclose"；如系统坚持要文件则传 `conflict_of_interest.pdf` | 正文 Acknowledgment 已含 COI 声明 |

### ⚪ Optional Files（按适用上传）

| ScholarOne 槽位 | 上传文件 | 说明 |
|---|---|---|
| **Supplementary Material for Review** | `supplementary_material.pdf` + `supplementary.zip` | 表格/图/算法的补充 PDF（**用 Overleaf XeLaTeX 重编 `supplementary_material.tex` 获得 Noto 字体版**）+ 数据/音频样本压缩包 |
| **Previously Published – Statement** | `description_of_differences.pdf` | Summary of Differences（vs ICASSP 2026）|
| **Previously Published – Files** | `icassp2026_paper.pdf` | ICASSP 2026 会议论文（可再附 `original_presentation_icassp2026.pdf` 幻灯片）|
| **Cover letter / Comments** | `cover_letter.pdf` | **仅 Editor 可见**，含作者变更说明 |
| **LaTeX Supplementary File** | `supplementary_latex.zip` | 补充材料源文件（`supplementary_material.tex` + `fig_2.png`）|
| **Image**（可选）| `fig_1_v2.png` | 已嵌在 Main Manuscript 内，可不重复上传 |

**可跳过的槽位**（初投不适用）：Main Document – Tracked Changes（修回才需要）、Additional File for Review but Not for Publication、Supplementary Material Not for Review、Supporting Document（被拒后回复审稿人才需要）。

> ⚠️ Cover letter 槽位（"Cover letter / Comments"）不对 reviewer 显示——作者变更说明等放这里。
> ⚠️ Main Manuscript 上传后**务必下载它生成的 review proof PDF**，确认中日文示例（缘→縁 等）正常显示、无编译报错；若 ScholarOne 默认 pdfLaTeX 导致 CJK 报错，联系我出 pdfLaTeX 兼容版。

---

## main.pdf 合规清单(submit 前要逐项核对)

- [ ] **用 XeLaTeX 编译**(含 `xeCJK`，pdfLaTeX 会失败);中日文示例正常显示
- [ ] `\documentclass[journal]{IEEEtran}`
- [ ] 题首作者块、`\thanks{}` 单位/邮箱完整;**Qiming Bao 唯一第一作者、Jing An 通讯作者**(无共一标注)
- [ ] Abstract、IEEEkeywords 显示正常
- [ ] 系统架构图 (`fig_1_v2.png`) 用 `figure*` 跨双栏(宽度 `0.56\textwidth`)
- [ ] 参考文献由 `IEEEtran.bst` 编译;正文 **10 页**(含参考文献)
- [ ] 0 overfull box;自引 `an2026icassp` 以**正常作者署名**出现(single-blind,不 anonymize)
- [ ] 所有图表有 caption,公式编号连贯;**Algorithm 1 在补充材料**(supplementary)
- [ ] COI 声明在 Acknowledgment;Reproducibility / Data Availability / Ethics 三段在文末

---

## supplementary.zip 内容(沿用 TOMM 版本,无须改动)

```
01_hyperparameters/   — LoRA YAML, Whisper config, vLLM launch script
02_prompts/           — OCR / ZS-translate / FT-translate prompts
03_evaluation/        — SacreBLEU signature, COMET checkpoint, jiwer, composite score
04_per_episode_metrics/ — placeholder CSVs for per-episode breakdowns
05_qualitative_examples/ — 50 translation triples + 30 OCR/ASR correction examples
06_dataset_metadata/  — episode statistics, fold splits, speaker inventory
07_compute/           — RTF profile per stage
README.md             — full index + reproduction recipe
```

---

## ⚠️ 投稿前还需要你做的事

### 1. 用 XeLaTeX 编译 main.pdf 和 supplementary_material.pdf

在 Overleaf(Compiler 选 **XeLaTeX**)或本地 TeX Live 编译。命令行:

```bash
xelatex main && bibtex main && xelatex main && xelatex main
xelatex supplementary_material   # 补充材料同样用 XeLaTeX,得到 Noto 字体版
```

> Main Manuscript 槽位本身上传的是 `main_manuscript.zip`(源文件,系统自编译);上面手动编译是为了**自己先核对 PDF 没问题**、以及拿到 Noto 字体版的 `supplementary_material.pdf`。

### 2. Cover letter 占位符替换

[cover_letter.md](cover_letter.md) 现有两处占位符,修改后重新生成 PDF:
- `[INSERT SUBMISSION DATE]` → 实际投稿日期
- `[INSERT NAMES IF ANY; LEAVE EMPTY OTHERWISE]` → Reviewer-excluded 名单,**没有就删掉整段**

```bash
cd /data/home/qbao775/translation/tmm_submission
python3 build_cover_letter.py
```

### 3. ORCID 等待(建议)

| 作者 | ORCID | 状态 |
|---|---|---|
| Jing An | — | ⬜ 等待 |
| Qiming Bao | `0000-0002-1000-7383` | ✅ |
| Rui-Yang Ju | `0000-0003-2240-1377` | ✅ |
| Haofei Chang | `0009-0003-6690-9065` | ✅ |
| Jinhua Su | `0000-0002-6344-0607` | ✅ |
| Yanbing Bai | `0000-0001-5223-9425` | ✅ |
| Xin Qu | — | ⬜ 等待 |

通讯作者已有 ORCID,**满足投稿门槛**;Jing An / Xin Qu 在接收前补即可。

---

## ScholarOne (TMM) 7 步流程速记

| Step | 内容 |
|---|---|
| 1 | Type=Regular Paper;Title 与 Running Head 复制 [main.tex](main.tex);Abstract 复制 abstract block(去掉 LaTeX 命令) |
| 2 | Keywords 9 个(从 `\begin{IEEEkeywords}` 直接搬);如系统要求 EDICS 选 1-2 个最贴近的(Multimedia Processing / Video Analysis 类) |
| 3 | 8 位作者按以下顺序录入,**只勾 Jing An 一人为 Corresponding Author**:`Qiming Bao, Jing An, Rui-Yang Ju, Haofei Chang, Jinhua Su, Yanbing Bai, Xin Qu, Michael Witbrock`(Qiming Bao 为唯一第一作者,Jing An 为通讯作者,作者顺序不变) |
| 4 | Preferred / Non-Preferred Reviewers 留空(除非有 COI) |
| 5 | 上传 `cover_letter.pdf`(Cover letter / Comments 槽位);填 **Conference Disclosure**(声明 ICASSP 2026 prior version,对应 `icassp2026_paper.pdf` 与 `description_of_differences.pdf`)/ COI(勾"无")/ Data Availability。**注意作者变更已在 cover letter 中说明**(Qiming Bao 为新增第一作者) |
| 6 | 按上方"ScholarOne 槽位映射"逐个槽位上传文件,**核对每个文件进对槽位** |
| 7 | 下载 review proof PDF 逐页核查(**中日文示例是否正常**) → 勾选合规声明 → Submit |

---

## 提交完成后立即做

1. 截图保存 Manuscript ID(形如 `TMM-XXXXX`)
2. 群发 7 位作者 Manuscript ID + ScholarOne 链接,提醒查收 co-author confirmation email
3. 追 Jing An / Xin Qu 的 ORCID(接收前必须补齐)

---

## 与 TOMM 投稿包的区别(对照表)

| 项 | TOMM 版本 (`tomm_submission/`) | TMM 版本 (本目录) |
|---|---|---|
| LaTeX class | `acmart` (`acmsmall, manuscript, screen, review, anonymous`) | `IEEEtran` (`journal`) |
| 评审模式 | Double-anonymous(双盲) | Single-blind(只对 reviewer 隐藏审稿意见,作者公开) |
| 作者匿名 | 在 `main.tex` 中删除/打码作者信息 | **保留**完整作者信息(`\thanks{}` 块) |
| 自引匿名 | `an2026icassp` 写成 "Anonymous Authors" | 保留正常作者署名 |
| Bibliography style | `ACM-Reference-Format.bst` | `IEEEtran.bst` |
| Keywords block | `\keywords{}` + CCS Concepts | `\begin{IEEEkeywords}` |
| Title page | `title_page.tex` (manuscript ID `TOMM-2026-0446`) | `title_page.tex` (manuscript ID 待 ScholarOne 分配) |
| Description of differences | ACM policy 30% threshold 描述 | IEEE policy "substantial extension" 描述,自引不再 anonymize |
| Cover letter | TOMM scope, ACM submission requirements | TMM scope, IEEE submission requirements |
| `main.pdf` 状态 | 已编译 | **需在 Overleaf 重新编译** |

如果 ScholarOne 任何一步报错,把错误信息发给我帮你排查。
