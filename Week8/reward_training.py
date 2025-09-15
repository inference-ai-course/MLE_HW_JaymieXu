from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import os

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "reward_data.jsonl")

    # Load tokenizer and model (following professor's code exactly)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", use_safetensors=True, num_labels=1)
    model.config.use_cache = False
    model.gradient_checkpointing_disable()

    # Load dataset from our reward_data.jsonl (always from script directory)
    dataset = load_dataset("json", data_files=data_file, split="train")

    # Preprocess function (following professor's code)
    MAX_LEN = 512
    def preprocess(batch):
        enc_ch = tokenizer(
            batch["chosen"], 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_LEN
        )
        enc_rj = tokenizer(
            batch["rejected"], 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_LEN
        )
        return {
            "input_ids_chosen": enc_ch["input_ids"],
            "attention_mask_chosen": enc_ch["attention_mask"],
            "input_ids_rejected": enc_rj["input_ids"],
            "attention_mask_rejected": enc_rj["attention_mask"],
        }

    # Apply preprocessing
    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    # Training arguments (using RewardConfig for RewardTrainer)
    training_args = RewardConfig(
        output_dir="reward_model",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_steps=10,
        fp16=True,
        remove_unused_columns=False,  # Keep this False for RewardTrainer
        gradient_checkpointing=False,  # Explicitly disable gradient checkpointing
    )

    # Create RewardTrainer (add tokenizer as processing_class)
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer
    )

    # Train the model
    trainer.train()

    # Explicitly save the final model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained("reward_model")

if __name__ == "__main__":
    main()