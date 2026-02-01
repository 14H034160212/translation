import asyncio
import edge_tts
import os
from pathlib import Path

async def generate_speech(text, voice, output_file):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

async def main():
    output_dir = Path("baseline_results/edgetts")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Using a standard Japanese voice
    voice = "ja-JP-NanamiNeural" 
    text = "これは私の履歴書です、どうぞご覧ください。"
    
    print("Generating EdgeTTS samples...")
    # Generate sample for "speakers"
    ref_dir = Path("data/tts_ref_audio")
    if ref_dir.exists():
        speakers = [d.name for d in ref_dir.iterdir() if d.is_dir()]
    else:
        speakers = [f"spk_{i}" for i in range(5)]

    for spk in speakers:
        out_file = output_dir / f"{spk}.mp3"
        await generate_speech(text, voice, str(out_file))
        print(f"Generated {out_file}")
    
    print("EdgeTTS processing complete.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop_policy().get_event_loop()
    loop.run_until_complete(main())
