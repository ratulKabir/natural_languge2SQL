# Generate SQL queries from Natural Language


## Setting up environment

Create a python 3.10 environment. Use the follwing command to install required libraries.

`pip install -r requirements.txt`

## Fine-Tuning the SQL Generator

This repository supports fine-tuning an LLM to generate SQL queries from natural language using LoRA and TRL's `SFTTrainer`.

### 🧪 Dataset

We use the [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) dataset. Each entry includes:
- A `sql_prompt`: natural language question
- A `sql`: corresponding SQL query

### ⚙️ Training Configuration

- **Model**: By default, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (can be changed in `config/config.py`)
- **LoRA**: Lightweight fine-tuning using PEFT
- **Trainer**: TRL's `SFTTrainer` with gradient checkpointing, cosine scheduler, and TensorBoard logging

### 📦 Setup

Make sure your `config/config.py` defines:
```python
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUT_DIR = "data/sql-sft-lora"
```

### 🔧 Fine-tune the Model

Run:
```bash
python fine_tune.py
```

This will:
1. Load and preprocess the dataset
2. Create the tokenizer and trainer
3. Start LoRA fine-tuning
4. Save the model to the output directory

### 📁 Output

- Model checkpoints are saved to `data/sql-sft-lora`
- TensorBoard logs are available for monitoring training progress
