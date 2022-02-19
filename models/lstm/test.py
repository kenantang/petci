import argparse
import torch as th
from torch.utils.data import DataLoader
import numpy as np

from lstm import LSTM
from dataloader import Dataset
from dataloader import batcher

import json
from os.path import exists

def main(args):
    device = th.device('cpu')

    testset = Dataset(mode=args.test_set)
    test_loader = DataLoader(dataset=testset,
                             batch_size=100,
                             collate_fn=batcher,
                             shuffle=True,
                             num_workers=0)

    model = LSTM(testset.vocab_size,
                 args.x_size,
                 args.h_size,
                 args.num_classes,
                 args.dropout,
                 pretrained_emb = testset.pretrained_emb).to(device)
    print(model)

    # test
    model.load_state_dict(th.load(args.directory + '/' + args.model))
    accs = []
    model.eval()
    for batch in test_loader:
        with th.no_grad():
            logits = model(batch)

        pred = th.argmax(logits, 1)
        acc = th.sum(th.eq(batch.label, pred)).item()
        accs.append([acc, len(batch.label)])

    correct = np.sum([x[0] for x in accs])
    total = np.sum([x[1] for x in accs])
    test_acc = 1.0*correct/total

    # print result as json
    result = {}

    result["model"] = args.model
    result["test-set"] = args.test_set
    result["correct"] = int(correct)
    result["total"] = int(total)
    result["accuracy"] = round(test_acc, 4)

    print(result)

    # save to file
    with open(args.summary, "a") as f:
        f.write(json.dumps(result)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x-size', type=int, default=300)
    parser.add_argument('--h-size', type=int, default=168)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--test-set', type=str, default="test-g")
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--directory', type=str, default=None)
    parser.add_argument('--summary', type=str, default='test_summary.jsonl')
    args = parser.parse_args()
    print(args)
    assert args.model is not None, "No model provided!"
    assert args.directory is not None, "Directory not specified!"
    assert exists(args.directory + '/' + args.model), "Model does not exist!"
    main(args)