# IEEE TCSVT 投稿包

**Manuscript:** *An End-to-End Multimodal System for Subtitle Recognition and Chinese–Japanese Translation with Speech Synthesis in Short Dramas*
**Submission system:** ScholarOne Manuscripts — <https://mc.manuscriptcentral.com/tcsvt>
**Generated:** 2026-05-09 (TCSVT version, ported from `tomm_submission/`)

> ⚠️ TCSVT 评审默认是 **single-blind**(reviewer 看不到作者,但作者署名在投稿稿件上)。本投稿包不再像 TOMM 版本那样做匿名化:`main.tex` 顶部已经写入完整作者名、单位与邮箱(IEEEtran `\thanks{}` 块),自引也以正常作者格式呈现。投稿前请按下方"main.pdf 合规清单"再核对一遍。

---

## 文件清单（按上传顺序）

| 顺序 | 文件 | 大小 (字节) | ScholarOne File Type | 谁能看到 |
|---|---|---|---|---|
| 1 | `main.pdf` *(由 `main.tex` 编译产生)* | TBD | **Main Document** | Editor + Reviewers |
| 2 | `main.tex` | 60 KB | LaTeX Source | Editor + Reviewers |
| 3 | `refs.bib` | 15 KB | LaTeX Source | Editor + Reviewers |
| 4 | `fig_1_v2.png` | 445 KB | Figure | Editor + Reviewers |
| 5 | `fig_2.png` | 48 KB | Figure | Editor + Reviewers |
| 6 | `original_presentation_icassp2026.pdf` | 2.2 MB | **Prior Conference Paper** | Editor (often visible to Reviewers) |
| 7 | `description_of_differences.pdf` | 7 KB | **Summary of Differences** | Editor + Reviewers |
| 8 | `supplementary.zip` | 18 KB | **Supplementary Material** | Editor + Reviewers |
| 9 | `cover_letter.pdf` | 7 KB | **Cover Letter** | **Editor only** |
| 10 | `title_page.pdf` *(可选, 见说明)* | 2 KB | Title Page | Editor only |

> ⚠️ Cover letter 上传时务必把 file type 设为 "Cover Letter",否则会进 reviewer 包。
> ⚠️ ScholarOne 上 IEEE 用的 file-type 选项随会随期刊微调,按现场标签选最贴近的,如 "Manuscript File" / "Bibliography" / "Image" 等。
> ⚠️ 本目录下的 `main.pdf` **本地未生成**(机器无 LaTeX),请用 Overleaf 或本地 TeX Live 编译 `main.tex` 后再上传第 1 项。

**Total**: 10 files, 约 2.8 MB,远低于 ScholarOne 单文件 30 MB / 总计 100 MB 上限。

---

## main.pdf 合规清单(submit 前要逐项核对)

- [ ] `\documentclass[journal]{IEEEtran}` (或 review-stage `[journal,onecolumn,11pt,draftcls]{IEEEtran}`)
- [ ] 题首作者块、`\thanks{}` 单位/邮箱、共一/通讯标注完整
- [ ] Abstract、IEEEkeywords 显示正常
- [ ] 系统架构图 (`fig_1_v2.png`) 用 `figure*` 跨双栏
- [ ] 参考文献由 `IEEEtran.bst` 编译
- [ ] 自引 `an2026icassp` 以**正常作者署名**出现(TCSVT 是 single-blind,不需要 anonymize)
- [ ] PDF 元数据中作者名段与 `main.tex` 一致
- [ ] 所有图都有 caption,公式编号连贯,Algorithm 1 显示正常

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

## ⚠️ 投稿前还需要你做的 3 件事

### 1. 编译 main.pdf

在 Overleaf 或本地 TeX Live 上把 [main.tex](main.tex) 编译出 `main.pdf`,放回本目录。命令行:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

### 2. Cover letter 占位符替换

[cover_letter.md](cover_letter.md) 现有两处占位符,修改后再重新生成 PDF:
- 第 13 行 `[INSERT SUBMISSION DATE]` → 实际投稿日期(例如 `May 15, 2026`)
- 第 49 行 `[INSERT NAMES IF ANY; LEAVE EMPTY OTHERWISE]` → 实际名单,**没有就把整段 Reviewer-excluded list 删掉**

```bash
cd /data/home/qbao775/translation/tcsvt_submission
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

## ScholarOne (TCSVT) 7 步流程速记

| Step | 内容 |
|---|---|
| 1 | Type=Regular Paper;Title 与 Running Head 复制 [main.tex](main.tex);Abstract 复制 abstract block(去掉 LaTeX 命令) |
| 2 | Keywords 9 个(从 `\begin{IEEEkeywords}` 直接搬);如系统要求 EDICS 选 1-2 个最贴近的(Multimedia Processing / Video Analysis 类) |
| 3 | 7 位作者按以下顺序录入,**Qiming Bao + Yanbing Bai 两个都勾 Corresponding Author**:`Jing An, Qiming Bao, Rui-Yang Ju, Haofei Chang, Jinhua Su, Yanbing Bai, Xin Qu` |
| 4 | Preferred / Non-Preferred Reviewers 留空(除非有 COI) |
| 5 | Cover letter 内容粘贴进文本框 *或* Step 6 上传 `cover_letter.pdf`;填 **Conference Disclosure**(声明 ICASSP 2026 prior version,引用 `original_presentation_icassp2026.pdf` 与 `description_of_differences.pdf`)/ COI / Data Availability |
| 6 | 按本文档"文件清单"顺序上传文件,**逐个核对 file type** |
| 7 | 下载 review proof PDF 逐页核查 → 勾选合规声明 → Submit |

---

## 提交完成后立即做

1. 截图保存 Manuscript ID(形如 `TCSVT-XX-XXXX`)
2. 群发 7 位作者 Manuscript ID + ScholarOne 链接,提醒查收 co-author confirmation email
3. 追 Jing An / Xin Qu 的 ORCID(接收前必须补齐)

---

## 与 TOMM 投稿包的区别(对照表)

| 项 | TOMM 版本 (`tomm_submission/`) | TCSVT 版本 (本目录) |
|---|---|---|
| LaTeX class | `acmart` (`acmsmall, manuscript, screen, review, anonymous`) | `IEEEtran` (`journal`) |
| 评审模式 | Double-anonymous(双盲) | Single-blind(只对 reviewer 隐藏审稿意见,作者公开) |
| 作者匿名 | 在 `main.tex` 中删除/打码作者信息 | **保留**完整作者信息(`\thanks{}` 块) |
| 自引匿名 | `an2026icassp` 写成 "Anonymous Authors" | 保留正常作者署名 |
| Bibliography style | `ACM-Reference-Format.bst` | `IEEEtran.bst` |
| Keywords block | `\keywords{}` + CCS Concepts | `\begin{IEEEkeywords}` |
| Title page | `title_page.tex` (manuscript ID `TOMM-2026-0446`) | `title_page.tex` (manuscript ID 待 ScholarOne 分配) |
| Description of differences | ACM policy 30% threshold 描述 | IEEE policy "substantial extension" 描述,自引不再 anonymize |
| Cover letter | TOMM scope, ACM submission requirements | TCSVT scope, IEEE submission requirements |
| `main.pdf` 状态 | 已编译 | **需在 Overleaf 重新编译** |

如果 ScholarOne 任何一步报错,把错误信息发给我帮你排查。
