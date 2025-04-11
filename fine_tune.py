from utils.dataprocess import get_formatted_dataset
from agents.sfttrainer_lora import create_sql_trainer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

train_dataset, eval_dataset, tokenizer = get_formatted_dataset(train_size=10000, test_size=100, 
                                                    model_id = MODEL_ID)

trainer = create_sql_trainer(MODEL_ID, tokenizer, train_dataset, eval_dataset, output_dir='data/sql-sft-lora')

train_result = trainer.train()