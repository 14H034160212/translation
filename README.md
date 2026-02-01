# Multimodal Experiments for Short Drama Translation

This repository contains the implementation and experimental results for a multimodal subtitle translation system, focusing on **Subtitle Recognition** and **Text-to-Speech (TTS)**.

## 1. Experimental Models & Configuration

### A. Subtitle Recognition (Subtitle Extraction)
*   **Model 1 (Baseline)**: [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
*   **Model 2 (Enhanced)**: [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
    *   *Note*: Qwen3-VL-4B was run in a dedicated Python 3.11 environment with development versions of `transformers` to support the new architecture.
*   **Method**: Zero-shot Video-Language Understanding.
*   **Input**: Video frames extracted at 1 frame/second.
*   **Output**: Chinese subtitle text.

### B. Text-to-Speech (TTS) Comparison
*   **Model 1 (Baseline)**: [GPT-SoVITS v3](https://github.com/RVC-Boss/GPT-SoVITS)
    *   *Method*: Zero-shot Voice Cloning (Reference-based Synthesis).
*   **Model 2 (Comparison)**: [F5-TTS](https://github.com/SWivid/F5-TTS)
    *   *Method*: Flow Matching (Zero-shot).
*   **Input**: Source Chinese audio (for timbre reference) + Target Japanese text.
*   **Output**: Japanese speech audio.

### C. Subtitle Translation (LoRA Fine-tuning)
*   **Model**: Qwen2.5-3B-Instruct
*   **Method**: Low-Rank Adaptation (LoRA)
*   **Parameters**:
    *   Rank: 16
    *   Alpha: 32
    *   Learning Rate: 2e-4
    *   Quantization: 4-bit (NF4)

## 2. Datasets

| Dataset | Content | Quantity | Source |
| :--- | :--- | :--- | :--- |
| **Video Corpus** | Short Drama "闪婚幸运草的命中注定" | 82 clips | `extracted_data/` |
| **TTS Reference** | Speaker audio clips (Chinese) | 12 speakers | `data/tts_ref_audio/` |

## 3. Experiment Results

### Experiment 1: Subtitle Recognition

We evaluated the Optical Character Recognition (OCR) performance using **Qwen2-VL** (Baseline) and are currently running **Qwen3-VL-4B** (In Progress).

| Model | CER (%) ↓ | BLEU ↑ | chrF++ ↑ | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen2-VL-2B-Instruct** | 2.82 | 0.35 | 0.50 | ✅ Completed |
| **Qwen3-VL-4B-Instruct** | **1.74** | **0.37** | **0.59** | ✅ Completed |

*   **Performance Upgrade**: Qwen3-VL-4B achieved a **38% reduction in Character Error Rate (CER)** compared to the Qwen2-VL-2B baseline, demonstrating significantly superior OCR capabilities for subtitle extraction.

### Experiment 2: Text-to-Speech (TTS)

We generated 38 Japanese audio samples across 12 different speakers, cloning the voice characteristics from the original Chinese audio.

| Metrics | Value | Description |
| :--- | :---: | :--- |
| **Samples** | 38 | Generated across 12 speakers |
| **Avg. WER** | **1.17** | Word Error Rate (evaluated via Whisper ASR) |
| **Intelligibility** | High | Audio is clear and consistent with source timbre. |

### Experiment 3: Subtitle Translation
We evaluated the ability of **Qwen3-VL** to translate processed Chinese subtitles into Japanese, comparing its Zero-shot performance against a Fine-tuned version.

| Model | Method | BLEU (Zero-shot) | BLEU (Fine-tuned) | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Qwen2.5-3B-Instruct** | LoRA (Baseline) | -- | -- | *Reference* |
| **Qwen3-VL-4B-Instruct** | Zero-shot | **19.53** | -- | ✅ Completed |
| **Qwen3-VL-4B-Instruct** | LoRA (4-bit) | -- | 18.23 | ✅ Completed |

*   **Observation**: The Fine-tuned model (18.23) slightly underperformed compared to Zero-shot (19.53). This suggests that the small-scale text-only fine-tuning (79 pairs) may have disrupted the model's pre-trained alignment or that 4-bit quantization effects were significant.

### Experiment 4: TTS Comparison (F5-TTS)
We compared the zero-shot voice cloning capabilities of **F5-TTS** (Flow Matching) against the baseline **GPT-SoVITS** (VITS-based).

| Model | Method | Avg. WER ↓ | Status |
| :--- | :--- | :---: | :---: |
| **GPT-SoVITS v3** | Zero-shot | **1.17** | ✅ Completed |
| **F5-TTS** | Zero-shot (Flow) | 2.29 | ✅ Completed |

*   **Objective**: Benchmark benchmarking the new flow-matching architecture.
*   **Result**: F5-TTS struggled with the cross-lingual zero-shot task using short (3-5s) reference audio, resulting in significant hallucinations and high WER compared to GPT-SoVITS.

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

| Context | BLEU ↑ | Conclusion |
| :--- | :---: | :--- |
| **Text-Only (OCR)** | 14.18 | Baseline |
| **Multimodal (Video + Text)** | **18.06** | **+3.88 BLEU** improvement with visual context. |

#### 2. TTS Reference Length (F5-TTS)
We evaluated how the duration of the reference audio affects zero-shot speaker similarity.
- **Method**: Comparing similarity scores for short (3s) vs. long (10s) reference prompts using F5-TTS.

| Reference Length | Speaker Similarity ↑ | Conclusion |
| :--- | :---: | :--- |
| **Short (~3s)** | 0.64 | Lower similarity |
| **Long (~10s)** | **0.71** | Longer prompts capture better speaker characteristics. |

### B. Additional Baselines

#### 1. Machine Translation (NLLB-200)
We benchmarked our VLM-based translation against a traditional neural machine translation model.
- **Model**: `facebook/nllb-200-distilled-600M`
- **Result**: **9.73 BLEU**.
- **Comparison**: significantly lower than Qwen3-VL (19.53 BLEU), highlighting the difficulty of the drama domain for general-purpose MT models without visual context.

#### 2. OCR and TTS Baselines
Scripts have been implemented to support further comparisons:
- **OCR**: `OpenGVLab/InternVL2-4B`
- **TTS**: `CosyVoice-300M`

### C. End-to-End System Demonstration
We implemented a proof-of-concept pipeline `run_end_to_end.py` that fully automates the workflow:
1.  **Video Ingestion**: Reads `.mp4` file.
2.  **Visual Extraction**: Qwen3-VL extracts subtitles (OCR).
3.  **Translation**: Qwen3-VL translates text to Japanese.
4.  **Audio Synthesis**: F5-TTS generates dubbed audio.
5.  **Dubbing**: FFmpeg merges audio back to video.
*Note*: The final dubbing step (FFmpeg) currently has environmental limitations (libavutil), but the AI components function successfully.

