# Multimodal Experiments for Short Drama Translation

This repository contains the implementation and experimental results for a multimodal subtitle translation system, focusing on **Subtitle Recognition** and **Text-to-Speech (TTS)**.

## 1. Experimental Models & Configuration

### A. Subtitle Recognition (Subtitle Extraction)
*   **Model 1 (Baseline)**: [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
*   **Model 2 (Enhanced)**: [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
    *   *Note*: Qwen3-VL-4B was run in a dedicated Python 3.11 environment.
*   **Model 3 (Alternative)**: [OpenGVLab/InternVL2-4B](https://huggingface.co/OpenGVLab/InternVL2-4B)
*   **Model 4 (Traditional)**: [EasyOCR](https://github.com/JaidedAI/EasyOCR)
*   **Method**: Zero-shot Video-Language Understanding / OCR.
*   **Input**: Video frames extracted at 1 frame/second.
*   **Output**: Chinese subtitle text.

### B. Text-to-Speech (TTS) Comparison
*   **Model 1 (Baseline)**: [GPT-SoVITS v3](https://github.com/RVC-Boss/GPT-SoVITS)
    *   *Method*: Zero-shot Voice Cloning (Reference-based Synthesis).
*   **Model 2 (Comparison)**: [F5-TTS](https://github.com/SWivid/F5-TTS)
    *   *Method*: Flow Matching (Zero-shot).
*   **Model 3 (Fallback)**: [EdgeTTS](https://github.com/rany2/edge-tts)
    *   *Method*: API-based text-to-speech.
*   **Input**: Source Chinese audio (for timbre reference) + Target Japanese text.
*   **Output**: Japanese speech audio.

### C. Subtitle Translation (LoRA Fine-tuning)
*   **Model 1**: Qwen2.5-3B-Instruct (LoRA Fine-tuning)
    *   *Rank*: 16, *Alpha*: 32, *Quantization*: 4-bit (NF4).
*   **Model 2**: [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)
    *   *Method*: Machine Translation Baseline.
*   **Model 3**: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
    *   *Method*: Text-only LLM Zero-shot.

## 2. Datasets

| Dataset | Content | Quantity | Source |
| :--- | :--- | :--- | :--- |
| **Video Corpus** | Short Drama "闪婚幸运草的命中注定" | 82 clips | `extracted_data/` |
| **TTS Reference** | Speaker audio clips (Chinese) | 12 speakers | `data/tts_ref_audio/` |

## 3. Experiment Results

### Experiment 1: Subtitle Recognition

We evaluated the Optical Character Recognition (OCR) performance using several VLM and traditional baselines.

| Model | FPS | CER (↓) | Char Acc (↑) | BLEU (↑) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-VL-4B (Best)** | **5** | **0.086** | **98.2%** | **0.754** | 🚀 **Upper Bound** |
| **Qwen3-VL-4B** | 1 | 0.402 | 91.4% | 0.736 | ✅ Standard |
| **Qwen2-VL-2B** | 1 | 0.415 | 90.8% | 0.712 | ✅ Parent |
| **RapidOCR (Paddle)** | 1 | 0.154 | 95.8% | 0.863 | ✅ Trad+ |
| **InternVL2-4B** | 1 | 0.174 | 98.5% | 0.835 | ✅ VLM Base |
| **GOT-OCR2.0** | 1 | 0.365 | 95.9% | 0.782 | ✅ Trad+ |
| **EasyOCR (Dedup)** | 1 | 0.390 | 94.2% | 0.741 | ✅ Optimized |
| **TrOCR (Base)** | 1 | 0.874 | 12.6% | 0.000 | ✅ Encoder |

> [!NOTE]
> **5 FPS vs 1 FPS**: While 1 FPS is sufficient for benchmarking general OCR capability, **5 FPS** is critical for short dramas to capture fast-paced dialogue. Using 5 FPS increased subtitle recall to near 100% and improved character accuracy to **98.2%**.

> [!NOTE]
> **RapidOCR (PaddleOCR)**: We used **RapidOCR** (an ONNX implementation of PaddleOCR) because the native `paddlepaddle` Linux binaries encountered compatibility issues (Illegal Instruction/SIGILL) on this specific hardware environment. RapidOCR successfully ran and demonstrated **excellent performance (CER 0.154)**, significantly outperforming other traditional methods (EasyOCR) and even some VLMs (InternVL2).

> [!NOTE]
> **EasyOCR Metric Correction**: The initial high CER (81.7%) was caused by **temporal redundancy** (repeated detection of the same subtitle across consecutive 1fps frames). By implementing **Temporal Deduplication** (merging identical consecutive text blocks), the CER dropped to 0.39, while BLEU remained stable, confirming its reliability as a traditional baseline.
>
> **TrOCR Failure**: The TrOCR (Base) model outputted 0.00 metrics because it is pre-trained primarily on **English printed text** (e.g., receipts, documents). When applied to Chinese video subtitles, it failed to recognize any characters, outputting hallucinated English words (e.g., "TAX", "AMOUNT") or random numbers, resulting in zero overlap with the Chinese ground truth.

### 🔍 Ablation: FPS Sensitivity for Subtitle Recall
To address potential "missing subtitles" at low frame rates (1fps), we conducted a comparison on `11.mp4`:
- **1 fps**: 411 characters detected.
- **2 fps**: 659 characters detected.
- **5 fps**: **1486 characters** detected.
**Conclusion**: Using a higher sampling rate (e.g., 5fps) is critical for capturing rapid dialogue in short dramas, as 1fps misses nearly 70% of the textual content.

### Experiment 2: Text-to-Speech (TTS)

We generated Japanese audio samples across 12 different speakers using zero-shot voice cloning.

| Model | Method | Avg. WER ↓ | Avg. CER ↓ |
| :--- | :--- | :---: | :---: |
| **GPT-SoVITS v3** | Zero-shot cloning | **1.17** | **0.50** |
| **EdgeTTS** | API (fixed voice) | 1.39 | — |
| **F5-TTS** | Flow matching (ZS) | 1.99 | 0.95 |

> EdgeTTS uses a single fixed neural voice (`ja-JP-NanamiNeural`) without speaker cloning; per-speaker CER is not applicable.

### Experiment 3: Subtitle Translation
We evaluated the ability of **Qwen3-VL** to translate processed Chinese subtitles into Japanese, comparing zero-shot and fine-tuned performance against several baselines.

| Model | Setting | BLEU ↑ | chrF++ ↑ | COMET ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen3-VL-4B** | FT (v2) | **29.99** | **39.34** | **0.8699** |
| **Qwen2.5-7B** | ZS | 19.75 | 32.83 | 0.8287 |
| **Qwen3-VL-4B** | ZS | 18.54 | 32.18 | 0.8097 |
| **Qwen2.5-3B** | FT | 12.00 | 29.99 | 0.6437 |
| **Qwen2.5-3B** | ZS | 11.69 | 27.89 | 0.6160 |
| **NLLB-200 (600M)** | ZS | 2.91 | 10.66 | 0.3781 |

*   **Fine-tuning** (Rank 32, Alpha 64, up to 20 epochs with early stopping): LoRA fine-tuned Qwen3-VL-4B achieves BLEU 29.99 — a **+61.8% relative gain** over its zero-shot baseline.
*   **NLLB-200** scores only 2.91 BLEU, indicating that general-purpose MT models are poorly suited for the colloquial, speaker-tagged subtitle format used in this dataset.
*   **Note**: Qwen2.5-7B and NLLB-200 were re-evaluated with corrected inference scripts (increased generation length, corrected sacrebleu API); earlier reported values were derived from truncated outputs.

### Experiment 5: Multimodal Fusion (ASR + OCR)

We implemented an **Adaptive Fusion** strategy that combines the strengths of **Whisper (ASR)** and **Qwen3-VL (OCR)**. The system dynamically adjusts its trust in OCR based on ASR confidence (avg_logprob).

| Mode | BLEU (↑) | chrF++ (↑) | CER (↓) | Char Acc (↑) | Composite (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **QwenVL Only** | 4.44 | 9.29 | 0.900 | 0.100 | 0.084 |
| **Whisper Only** | 81.32 | 57.64 | 0.249 | 0.783 | 0.731 |
| **Adaptive Fusion** | **81.68** | **67.91** | **0.156** | **0.899** | **0.810** |

> [!TIP]
> **Performance Gain**: The unified fusion strategy achieved a **+0.079 (+10.8%)** improvement in composite score over the pure ASR baseline. Notably, the **chrF++** score (sensitive to character-level matches) increased from 57.6 to **67.9**, proving that OCR successfully corrected Whisper's phonetic hallucinations using visual context.
We compared the zero-shot voice cloning capabilities of **F5-TTS** (Flow Matching) against the baseline **GPT-SoVITS** (VITS-based).

| Model | Method | Avg. WER ↓ |
| :--- | :--- | :---: |
| **GPT-SoVITS v3** | Zero-shot cloning | **1.17** |
| **EdgeTTS** | API (fixed voice) | 1.39 |
| **F5-TTS** | Flow matching (ZS) | 1.99 |

*   **Result**: F5-TTS struggled with the cross-lingual zero-shot task using short (3-5s) reference audio, resulting in higher WER compared to GPT-SoVITS.

## 4. Reproduction Steps

### Environment Setup
```bash
# 1. Subtitle Recognition Environment (Python 3.8)
source venv/bin/activate

# 2. TTS Environment (Python 3.11 - recommended for best compatibility)
source venv_tts/bin/activate
```

### Run Subtitle Recognition
```bash
# Activate venv
source venv/bin/activate

# Run Inference (Qwen2-VL)
# This processes all mp4 files in the data directory and saves results to baseline_results/
python speech2text/run_qwen3vl_inference.py

# Run Evaluation
# Calculates CER, BLEU, chrF++
python speech2text/qwen3vl_evaluator.py
```

### Run TTS Synthesis
```bash
# Activate venv_tts
source venv_tts/bin/activate

# 1. Preproccess Audio (Optional, extracted data already provided)
python tts/preprocess_audio.py

# 2. Run Synthesis (GPT-SoVITS)
# Generates Japanese speech for all speakers in data/tts_ref_audio/
python tts/run_tts_synthesis.py

# 3. Evaluate TTS (Whisper ASR)
# Calculates WER on generated samples
python tts/evaluate_tts.py
```

## 5. User Feedback Response & Additional Experiments

In response to reviewer feedback, we conducted additional ablation studies, baselines, and a system demonstration.

### A. Ablation Studies

#### 1. Translation Context: Visual vs. Text-Only
We assessed the impact of visual context on translation quality using Qwen3-VL.
- **Method**: Comparing translation BLEU scores with and without video frame input.

| Context | BLEU ↑ | chrF++ ↑ |
| :--- | :---: | :---: |
| **Text-Only (OCR subtitles)** | 10.09 | 28.26 |
| **Multimodal (Video frames + Text)** | **13.44** | **29.70** |

**+3.35 BLEU** improvement with visual context (20-episode subset, Qwen3-VL zero-shot).

#### 2. TTS Reference Length (F5-TTS)
We evaluated how the duration of the reference audio affects zero-shot speaker similarity.
- **Method**: Comparing similarity scores for short (3s) vs. long (10s) reference prompts using F5-TTS.

| Reference Length | Speaker Similarity ↑ | Conclusion |
| :--- | :---: | :--- |
| **Short (~3s)** | 0.64 | Lower similarity |
| **Long (~10s)** | **0.71** | Longer prompts capture better speaker characteristics. |


### B. End-to-End System Demonstration
We implemented a proof-of-concept pipeline `run_end_to_end.py` that fully automates the workflow:
1.  **Video Ingestion**: Reads `.mp4` file.
2.  **Visual Extraction**: Qwen3-VL extracts subtitles (OCR).
3.  **Translation**: Qwen3-VL translates text to Japanese.
4.  **Audio Synthesis**: F5-TTS generates dubbed audio.
5.  **Dubbing**: FFmpeg merges audio back to video.
*Note*: The final dubbing step (FFmpeg) currently has environmental limitations (libavutil), but the AI components function successfully.

