from utils.dataprocess import get_formatted_dataset
from agents.sfttrainer_lora import create_sql_trainer
from config.config import MODEL_ID, OUT_DIR

train_dataset, eval_dataset, tokenizer = get_formatted_dataset(train_size=10000, test_size=100, 
                                                                model_id=MODEL_ID)

trainer = create_sql_trainer(MODEL_ID, tokenizer, train_dataset, eval_dataset, output_dir=OUT_DIR)

train_result = trainer.train()