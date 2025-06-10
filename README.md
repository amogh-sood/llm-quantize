

---

LLM Quantization Comparator

A terminal-based CLI tool for evaluating and comparing large language models from Hugging Face across different quantization formats:

* Base (fp16)
* 8-bit quantized
* 4-bit quantized

The tool measures model size, perplexity, and perplexity efficiency per memory usage — helping you identify the best tradeoff between performance and resource consumption.

---

Quickstart


1. Clone and Run

```bash
git clone https://github.com/amogh-sood/llm-quantize
cd llm-quantize
python main.py
```

---

Features

* Compares memory footprint of each quantized model
* Calculates perplexity on a fixed test string
* Evaluates efficiency: perplexity per GB
* Displays results using rich-formatted tables
* Final recommendation printed based on overall efficiency
* Local caching of models under `models/`

---

Hugging Face Token

To run this tool, you need a Hugging Face token.

* Paste it when prompted at runtime.

If you’re using gated models (e.g. `google/medgemma-4b-it`), make sure you’ve accepted the terms on the model page.

---

Output Directory

After running the tool, the following directory structure is created:

```
models/
├── base/
├── int8/
└── 4bit/
```

Each folder contains the downloaded or quantized model files.

---

Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Required packages include:

* transformers
* bitsandbytes
* huggingface\_hub
* torch
* rich
* python-dotenv

---

Tested Models

These model IDs are known to work with this tool:
(2x Nvidia H100)

* google/gemma-2b-it
* Qwen/Qwen1.5-0.5B
* mistralai/Mistral-7B-Instruct-v0.1
* tiiuae/falcon-7b-instruct
* google/gemma-3-12b-it-qat-q4_0-unquantized
* microsoft/phi-4 

---

License

MIT

---

Author

Made by Amogh Sood
[https://github.com/amogh-sood](https://github.com/amogh-sood)

---

