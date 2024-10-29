from transformers import TrainingArguments, Trainer

def main():
    args = TrainingArguments(
        "test-ner",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    print("TrainingArguments created successfully")

if __name__ == "__main__":
    main()