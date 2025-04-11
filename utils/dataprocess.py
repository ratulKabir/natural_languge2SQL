from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from multiprocessing import cpu_count

def get_formatted_dataset(train_size=10000, test_size=100, model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    # === Load Dataset ===
    dataset = load_dataset("gretelai/synthetic_text_to_sql")

    # === Select subset ===
    dataset_dict = {
        "train": dataset["train"].select(range(train_size)),
        "test": dataset["test"].select(range(test_size))
    }
    raw_datasets = DatasetDict(dataset_dict)

    # === Format Examples ===
    def format_example(example):
        prompt = f"""### Instruction:
                {example['sql_prompt']}

                ### Response:
                """
        response = example["sql"]
        return {"text": prompt + response}

    raw_datasets = raw_datasets.map(format_example, remove_columns=raw_datasets["train"].column_names)

    # === Tokenization ===
    # model_id = "mistralai/Mistral-7B-v0.1" # too big
    model_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    # === Dataset Split ===

    def tokenize(example):
        return tokenizer(example["text"], truncation=True)

    raw_datasets = raw_datasets.map(tokenize, num_proc=cpu_count(), remove_columns=["text"])

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    return train_dataset, eval_dataset, tokenizer
