import argparse
from transformers import BertForSequenceClassification, Trainer
from transformers.trainer_utils import set_seed
import numpy as np
from dataloader import Dataset
import json

def main(args):
    set_seed(args.seed)

    test_dataset = Dataset(args.test_set)

    model_path = "./checkpoints/" + args.model
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=args.num_classes) 

    # Define test trainer
    test_trainer = Trainer(model)

    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset) 
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)
    y = test_dataset.labels
    correct = np.sum(np.equal(y_pred, y))
    total = len(y)
    test_acc = 1.0*correct/total

    # print result as json
    result = {}

    result["model"] = args.model
    result["test-set"] = args.test_set
    result["correct"] = int(correct)
    result["total"] = int(total)
    result["accuracy"] = round(test_acc, 4)

    print(result)

    if args.save:
        result["model"] = "best_{}_train-{}-{}_dev-{}.pkl".format(
            str(args.seed),
            args.hm,
            str(args.part),
            args.hm
        )
        with open(args.summary, "a") as f:
            f.write(json.dumps(result)+'\n')
    
    else:
        # print to a temporary file
        with open("check_points_acc.jsonl", "a") as f:
            f.write(json.dumps(result)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--hm', type=str, default="gh")
    parser.add_argument('--part', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--test-set', type=str, default="test-all")
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--summary', type=str, default='test_summary.jsonl')
    args = parser.parse_args()

    assert args.model is not None, "No model provided!"
    
    main(args)