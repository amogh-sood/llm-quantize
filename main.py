from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# load & quantize in one go
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    quantization_config=bnb_config,
    device_map="auto"  # or model.to("cuda")
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# save
model.save_pretrained("./quantized-qwen3-4b")
tokenizer.save_pretrained("./quantized-qwen3-4b")