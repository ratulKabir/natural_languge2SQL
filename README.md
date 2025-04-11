# Generate SQL queries from Natural Language


## Setting up environment

Create a python 3.10 environment. Use the follwing command to install required libraries.

`pip install -r requirements.txt`

## Fine-Tuning the SQL Generator

This repository supports fine-tuning an LLM to generate SQL queries from natural language using LoRA and TRL's `SFTTrainer`.

##### Testing the Fine-Tuned Model

After fine-tuning, you can test the model's performance on natural language to SQL generation.

### Inference Script

The `test.py` script loads:
- The base model defined in `config/config.py`
- The fine-tuned LoRA adapter from `data/sql-sft-lora/checkpoint-1250`

### Run Inference

To test the model with a custom prompt:

```bash
python test.py
```

The script uses the following example:

```text
Prompt: Show all users with gender is female.
```

It returns the generated SQL query based on your fine-tuned model.

### Notes

- The script uses `top_p` sampling and beam search for diversity and quality.
- Results are printed to stdout.
 Dataset

We use the [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) dataset. Each entry includes:
- A `sql_prompt`: natural language question
- A `sql`: corresponding SQL query

### Training Configuration

- **Model**: By default, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (can be changed in `config/config.py`)
- **LoRA**: Lightweight fine-tuning using PEFT
- **Trainer**: TRL's `SFTTrainer` with gradient checkpointing, cosine scheduler, and TensorBoard logging

### Setup

Make sure your `config/config.py` defines:
```python
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUT_DIR = "data/sql-sft-lora"
```

### Fine-tune the Model

Run:
```bash
python fine_tune.py
```

This will:
1. Load and preprocess the dataset
2. Create the tokenizer and trainer
3. Start LoRA fine-tuning
4. Save the model to the output directory

### Output

- Model checkpoints are saved to `data/sql-sft-lora`
- TensorBoard logs are available for monitoring training progress

## Deploy to Hugging Face Space

To upload the app, follow the `huggingFace_utils/upload2space.py` script. Mind that you need the proper HF token and your repo on HF for uploading.

The files which are necessary to run the app on teh HF space can found inside `huggingFace_utils/app`.

## Try the App yourself

* Follow the link https://huggingface.co/spaces/rat45/sql-sft-lora-model
* Enter a prompt like "Select all users where gender is male."
* In the output section you'll see the SQL query.

### Note

Since the model was trained only for an hour on a subset of the dataset. Some of the outputs will be imperfact.