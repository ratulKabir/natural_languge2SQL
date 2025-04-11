from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch

def create_sql_trainer(model_id, tokenizer, train_dataset, eval_dataset, output_dir='data/sql-sft-lora'):
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    # )

    device_map = {"": "mps"} if torch.backends.mps.is_available() else {"": "cpu"}

    model_kwargs = {
        "torch_dtype": 'auto',
        "use_cache": False,
        "device_map": device_map,
        # "quantization_config": quantization_config,
    }

    training_args = SFTConfig(
        # fp16=True,
        # do_eval=True,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2.0e-05,
        logging_steps=5,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_eval_batch_size=2,
        per_device_train_batch_size=2,
        save_strategy="epoch",
        seed=42,
        max_seq_length=tokenizer.model_max_length,
        report_to="tensorboard",
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    return trainer
