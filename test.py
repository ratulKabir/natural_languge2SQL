from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config.config import MODEL_ID

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "data/sql-sft-lora/checkpoint-1250")

# Inference
prompt = "Show all users with gender is female."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
        **inputs,
        max_new_tokens=128,  # speed things up
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True, 
        num_beams=5,
    )
output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output[len(prompt):].strip().split(';', 1)[0] + ';') 