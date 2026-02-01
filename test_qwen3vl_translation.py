import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_translation():
    base_model_id = "Qwen/Qwen3-VL-4B-Instruct" # Wait, Qwen3-VL 4B is a VL model.
    # For translation, we use the language model part.
    adapter_path = "translation/qwen3vl_translation_lora_v2_output/fold_0/final_model"
    
    print(f"🚀 Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"📦 Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    test_queries = [
        "Translate Chinese to Japanese: 宋氏集团的面试对我来说非常重要。",
        "Translate Chinese to Japanese: 我一定会查清楚母亲去世的真相。",
        "Translate Chinese to Japanese: 既然你这么想嫁进宋家，那我就成全你。"
    ]
    
    for query in test_queries:
        messages = [{"role": "user", "content": query}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=128,
                do_sample=False
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"\nQ: {query}")
            print(f"A: {response}")

if __name__ == "__main__":
    test_translation()
