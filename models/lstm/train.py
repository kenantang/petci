from ast import Pass
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse

from lstm import LSTM
from dataloader import Dataset
from dataloader import batcher

def main(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    best_epoch = -1
    best_dev_acc = 0

    # always use cpu
    device = th.device('cpu')

    trainset = Dataset(mode=args.train_set)
    train_loader = DataLoader(dataset=trainset,
                              batch_size=args.batch_size,
                              collate_fn=batcher,
                              shuffle=True,
                              num_workers=0)
    
    devset = Dataset(mode=args.dev_set)
    dev_loader = DataLoader(dataset=devset,
                            batch_size=100,
                            collate_fn=batcher,
                            shuffle=False,
                            num_workers=0)

    model = LSTM(trainset.vocab_size,
                 args.x_size,
                 args.h_size,
                 args.num_classes,
                 args.dropout,
                 pretrained_emb = trainset.pretrained_emb).to(device)
    print(model)

    # parameters that are not embedding
    params_ex_emb =[x for x in list(model.parameters()) if x.requires_grad and x.size(0)!=trainset.vocab_size]
    params_emb = list(model.embedding.parameters())

    for p in params_ex_emb:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)

    optimizer = optim.Adagrad([
        {'params':params_ex_emb, 'lr':args.lr, 'weight_decay':args.weight_decay},
        {'params':params_emb, 'lr':0.1*args.lr}])

    dur = []
    for epoch in range(args.epochs):
        t_epoch = time.time()
        model.train()
        for step, batch in enumerate(train_loader):
            if step >= 3:
                t0 = time.time()

            logits = model(batch)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, batch.label, reduction='sum')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step >= 3:
                dur.append(time.time() - t0)

            if step > 0 and step % args.log_every == 0:
                pred = th.argmax(logits, 1)
                acc = th.sum(th.eq(batch.label, pred))

                print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, step, loss.item(), 1.0*acc/len(batch.label), np.mean(dur)))
        print('Epoch {:05d} training time {:.4f}s'.format(epoch, time.time() - t_epoch))

        # eval on dev set
        accs = []
        model.eval()
        for step, batch in enumerate(dev_loader):
            with th.no_grad():
                logits = model(batch)

            pred = th.argmax(logits, 1)
            acc = th.sum(th.eq(batch.label, pred)).item()
            accs.append([acc, len(batch.label)])

        dev_acc = 1.0*np.sum([x[0] for x in accs])/np.sum([x[1] for x in accs])
        print("Epoch {:05d} | Dev Acc {:.4f}".format(epoch, dev_acc))

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch
            th.save(model.state_dict(),
                'checkpoints/best_{}_{}_{}.pkl'.format(args.seed, args.train_set,args.dev_set))
        else:
            # early stopping
            if best_epoch <= epoch - 10:
                break

        # lr decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(1e-5, param_group['lr']*0.99)
            print(param_group['lr'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--x-size', type=int, default=300)
    parser.add_argument('--h-size', type=int, default=168)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train-set', type=str, default="train-gh-1")
    parser.add_argument('--dev-set', type=str, default="dev-gh")
    args = parser.parse_args()
    print(args)
    main(args)