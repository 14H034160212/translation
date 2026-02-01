import os
import json
import torch
import glob
import subprocess
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
try:
    from transformers import Qwen3VLForConditionalGeneration
    MODEL_CLASS = Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForCausalLM
    MODEL_CLASS = AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
import sys

# Path to Translation Root
base_path = Path("/home/qbao775/translation")

class EndToEndPipeline:
    def __init__(self, output_dir="end_to_end_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Qwen3-VL (Vision, OCR, Translation)
        print("Loading Qwen3-VL...")
        self.model_path = "Qwen/Qwen3-VL-4B-Instruct" 
        
        # Using 4-bit to fit in memory
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16, 
            bnb_4bit_quant_type="nf4"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = MODEL_CLASS.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

    def extract_and_translate(self, video_path):
        """Unified Prompt: Extract Text AND Translate to Japanese."""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "fps": 1.0},
                    {"type": "text", "text": "Extract the Chinese subtitles from this video and provide a Japanese translation for each line. Format as: Timestamp - Chinese - Japanese."},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text

    def synthesize_audio(self, translation_text, speaker_ref_wav, output_path):
        """Call F5-TTS to generate audio from translation."""
        from f5_tts.infer.utils_infer import (
            infer_process,
            preprocess_ref_audio_text,
            load_vocoder,
            target_rms,
            cross_fade_duration,
            nfe_step,
            cfg_strength,
            sway_sampling_coef,
            speed,
            fix_duration,
            load_model
        )
        from omegaconf import OmegaConf
        from hydra.utils import get_class
        from cached_path import cached_path
        from importlib.resources import files
        
        # Load F5-TTS Model on demand (if not loaded)
        if not hasattr(self, "tts_model"):
             print("Loading F5-TTS...")
             
             model_name = "F5TTS_Base"
             vocoder_name = "vocos"
             model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
             model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
             model_arc = model_cfg.model.arch
             ckpt_file = str(cached_path(f"hf://SWivid/F5-TTS/{model_name}/model_1200000.safetensors"))
             
             self.tts_model = load_model(
                model_cls, model_arc, ckpt_file, mel_spec_type=vocoder_name, vocab_file="", device=self.device
             )
             self.tts_vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, device=self.device)
             
        # Ensure ref_wav exists
        if not Path(speaker_ref_wav).exists():
             print(f"Ref audio {speaker_ref_wav} not found.")
             # Fallback to a default if provided in CLI utils, but here we return
             return
             
        ref_text = "" 
        
        final_ref_audio, final_ref_text = preprocess_ref_audio_text(str(speaker_ref_wav), ref_text)
        
        audio, final_sr, spec = infer_process(
            final_ref_audio,
            final_ref_text,
            translation_text,
            self.tts_model,
            self.tts_vocoder,
            mel_spec_type="vocos",
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device
        )
        
        import soundfile as sf
        sf.write(output_path, audio, final_sr)

    def dub_video(self, video_path, audio_path, output_video_path):
        """Merge new audio with video using ffmpeg."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_video_path)
        ]
        print(f"Running ffmpeg: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

def run_pipeline(video_dir):
    pipeline = EndToEndPipeline()
    video_path = Path(video_dir) / "1.mp4"
    if not video_path.exists():
         print("Video 1.mp4 not found.")
         return
         
    # 1. Extract & Translate
    print("Extracting and Translating...")
    # For demo, result is "Timestamp - Chinese - Japanese"
    # We should parse Japanese text.
    result_text = pipeline.extract_and_translate(video_path)
    print(f"Result: {result_text}")
    
    # Simple parser: look for Japanese chars or just take the whole thing as TTS input?
    # Qwen3-VL might be chatty.
    # We will just pass the result to TTS for demo purposes, 
    # but practically we would parse the last part.
    # Let's clean it up slightly: remove lines starting with timestamp if possible.
    lines = result_text.split('\n')
    japanese_text = ""
    for line in lines:
        if " - " in line:
            parts = line.split(" - ")
            if len(parts) >= 3:
                japanese_text += parts[-1] + " "
        else:
            japanese_text += line + " " # Fallback
            
    if not japanese_text.strip():
        japanese_text = "こんにちは" # Fallback
    
    # 2. Synthesize
    print("Synthesizing Audio...")
    # Reference: Use concatenated short audio or existing file
    ref_dir = base_path / "data/tts_ref_audio/1"
    # Find a wav
    w = list(ref_dir.glob("*.wav"))
    if w:
        ref_wav = w[0]
    else:
        print("No ref wav found")
        return

    audio_out = pipeline.output_dir / "dubbed_audio.wav"
    pipeline.synthesize_audio(japanese_text, ref_wav, audio_out)
    
    # 3. Dub
    print("Dubbing Video...")
    final_video = pipeline.output_dir / "dubbed_video.mp4"
    pipeline.dub_video(video_path, audio_out, final_video)
    print(f"Finished! Saved to {final_video}")

if __name__ == "__main__":
    run_pipeline("extracted_data/闪婚幸运草的命中注定")
