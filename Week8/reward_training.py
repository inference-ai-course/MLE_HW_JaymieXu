from trl import RewardTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from datasets import load_dataset
import os

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "reward_data.jsonl")

    # Load tokenizer and model (following professor's code exactly)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=1)

    # Load dataset from our reward_data.jsonl (always from script directory)
    dataset = load_dataset("json", data_files=data_file, split="train")

    # Preprocess function (following professor's code)
    def preprocess(example):
        return tokenizer(example["chosen"], example["rejected"], truncation=True, padding="max_length")

    # Apply preprocessing
    dataset = dataset.map(preprocess, batched=True)

    # Training arguments (following professor's code)
    training_args = TrainingArguments(
        output_dir="reward_model",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_steps=10,
        fp16=True
    )

    # Create RewardTrainer (following professor's code exactly)
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    # Train the model
    trainer.train()

    # Explicitly save the final model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained("reward_model")

if __name__ == "__main__":
    main()