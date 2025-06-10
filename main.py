from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from dotenv import load_dotenv
import os
from huggingface_hub import login
from rich import print
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import time
import pandas as pd


console = Console()
load_dotenv()

# Function to convert bytes to gigabytes
bytes_to_gb = lambda num_bytes: num_bytes / (1024 ** 3)

# Function to calculate perplexity
def calculatePerplexity(model, text):
    encodings = tokenizer(text, return_tensors="pt").to("cuda")
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity

# Determine save paths relative to script
script_dir = os.path.dirname(os.path.abspath(__file__))
base_model_path = os.path.join(script_dir, "models", "base")
int8_model_path = os.path.join(script_dir, "models", "int8")
fourbit_model_path = os.path.join(script_dir, "models", "4bit")

os.makedirs(base_model_path, exist_ok=True)
os.makedirs(int8_model_path, exist_ok=True)
os.makedirs(fourbit_model_path, exist_ok=True)

# Ask for token
token = os.getenv("HF_TOKEN")
if not token:
    token = console.input("[bold cyan]Enter Hugging Face token (hidden): [/]")
if token:
    login(token=token)
else:
    raise ValueError("No token provided.")

# Model ID
modelID = console.input("[bold cyan]Enter model ID from Hugging Face:[/] ")

# Base model & tokenizer
with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
    task = progress.add_task(description="Loading base model...", total=None)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(modelID, cache_dir=base_model_path)
    baseModel = AutoModelForCausalLM.from_pretrained(modelID, cache_dir=base_model_path).to("cuda")
    baseModel.save_pretrained(base_model_path)
    tokenizer.save_pretrained(base_model_path)
    elapsed = time.time() - start
    progress.stop()
console.print(f"[green]Base model loaded in {elapsed:.2f} seconds[/]")

# 8-bit quantized model
bnb_config_int8 = BitsAndBytesConfig(load_in_int8=True)
with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
    task = progress.add_task(description="Loading 8-bit quantized model...", total=None)
    start = time.time()
    model_int8 = AutoModelForCausalLM.from_pretrained(
        modelID,
        cache_dir=int8_model_path,
        quantization_config=bnb_config_int8,
    )
    model_int8.save_pretrained(int8_model_path)
    elapsed = time.time() - start
    progress.stop()
console.print(f"[green]8-bit model loaded in {elapsed:.2f} seconds[/]")

# 4-bit quantized model
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
    task = progress.add_task(description="Loading 4-bit quantized model...", total=None)
    start = time.time()
    model_4bit = AutoModelForCausalLM.from_pretrained(
        modelID,
        cache_dir=fourbit_model_path,
        quantization_config=bnb_config_4bit,
    )
    model_4bit.save_pretrained(fourbit_model_path)
    elapsed = time.time() - start
    progress.stop()
console.print(f"[green]4-bit model loaded in {elapsed:.2f} seconds[/]")

# Function to get directory size
import os

def get_dir_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total

# Show model sizes
base_size = bytes_to_gb(get_dir_size(base_model_path))
int8_size = bytes_to_gb(get_dir_size(int8_model_path))
fourbit_size = bytes_to_gb(get_dir_size(fourbit_model_path))

table = Table(title="Model Size Comparison")
table.add_column("Model Type", style="bold cyan")
table.add_column("Size (GB)", justify="right")
table.add_row("Base (fp16)", f"{base_size:.2f}")
table.add_row("8-bit Quantized", f"{int8_size:.2f}")
table.add_row("4-bit Quantized", f"{fourbit_size:.2f}")
console.print(table)

# Perplexity evaluation
sample_text = '''Quantum computing represents a paradigm shift in computational science, leveraging principles of superposition and entanglement to solve problems intractable for classical computers. 
            While still in nascent stages of development, its potential applications span cryptography, drug discovery, and materials science. 
            However, engineering fault-tolerant quantum systems remains a formidable challenge, requiring advancements in qubit stability and error correction protocols. 
            The integration of quantum algorithms with classical pre- and post-processing techniques also presents complex architectural considerations for hybrid computing environments.'''


with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
    task = progress.add_task(description="Calculating base model perplexity...", total=None)
    start = time.time()
    base_perp = calculatePerplexity(baseModel, sample_text)
    elapsed = time.time() - start
    progress.stop()
console.print(f"[green]Base perplexity computed in {elapsed:.2f} seconds[/]")

with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
    task = progress.add_task(description="Calculating 8-bit model perplexity...", total=None)
    start = time.time()
    int8_perp = calculatePerplexity(model_int8, sample_text)
    elapsed = time.time() - start
    progress.stop()
console.print(f"[green]8-bit perplexity computed in {elapsed:.2f} seconds[/]")

with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
    task = progress.add_task(description="Calculating 4-bit model perplexity...", total=None)
    start = time.time()
    fourbit_perp = calculatePerplexity(model_4bit, sample_text)
    elapsed = time.time() - start
    progress.stop()
console.print(f"[green]4-bit perplexity computed in {elapsed:.2f} seconds[/]")

# Perplexity Table
perp_table = Table(title="Model Perplexity (Lower is Better)")
perp_table.add_column("Model Type", style="bold magenta")
perp_table.add_column("Perplexity", justify="right")
perp_table.add_row("Base (fp16)", f"{base_perp.item():.4f}")
perp_table.add_row("8-bit", f"{int8_perp.item():.4f}")
perp_table.add_row("4-bit", f"{fourbit_perp.item():.4f}")
console.print(perp_table)

# Calculate best model based on perplexity
# Compute ratios
base_ratio = base_perp.item() / base_size
int8_ratio = int8_perp.item() / int8_size
fourbit_ratio = fourbit_perp.item() / fourbit_size

# Extended Table: Efficiency
eff_table = Table(title="Model Efficiency (Perplexity / Size GB)")
eff_table.add_column("Model Type", style="bold yellow")
eff_table.add_column("Perplexity/GB", justify="right")
eff_table.add_row("Base (fp16)", f"{base_ratio:.4f}")
eff_table.add_row("8-bit", f"{int8_ratio:.4f}")
eff_table.add_row("4-bit", f"{fourbit_ratio:.4f}")
console.print(eff_table)

# Determine the most efficient model
best_ratio = min((base_ratio, "Base"), (int8_ratio, "8-bit"), (fourbit_ratio, "4-bit"))
console.print(f"\n[bold green]Most efficient model:[/] [cyan]{best_ratio[1]}[/] (lowest perplexity per GB)")

data = {
    "Model Type": ["Base (fp16)", "8-bit", "4-bit"],
    "Size (GB)": [base_size, int8_size, fourbit_size],
    "Perplexity": [base_perp.item(), int8_perp.item(), fourbit_perp.item()],
    "Perplexity/GB": [base_ratio, int8_ratio, fourbit_ratio],
}
df = pd.DataFrame(data)
df.attrs["title"] = modelID
df = pd.DataFrame(data)
csv_path = os.path.join(script_dir, "model_comparison.csv")
df.to_csv(csv_path, index=False)
console.print(f"[green]Data saved to {csv_path}[/]")
