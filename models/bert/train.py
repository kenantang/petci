import argparse
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import set_seed
from dataloader import Dataset

def main(args):
    set_seed(args.seed)
    
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=args.num_classes)

    train_dataset = Dataset(mode=args.train_set)
    dev_dataset = Dataset(mode=args.dev_set)

    training_args = TrainingArguments(
        output_dir='./checkpoints',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        seed=args.seed,
        save_total_limit=5,
        save_steps=args.save_steps,
        evaluation_strategy="no",
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--save-steps', type=int, default=500)
    parser.add_argument('--train-set', type=str, default="sst-train-tiny")
    parser.add_argument('--dev-set', type=str, default="sst-dev-tiny")
    args = parser.parse_args()
    print(args)
    main(args)