import argparse
import collections
import time
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl
from dataloader import Dataset

from tree_lstm import TreeLSTM

# skip warnings, may have side effects
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

Batch = collections.namedtuple('Batch', ['graph', 'mask', 'wordid', 'label', 'rootid'])

eps = 1e-30

def batcher(device):
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)

        rootid = [0]
        for b in batch:
            rootid.append(len(b)+rootid[-1])
        rootid.pop()

        return Batch(graph=batch_trees,
                     mask=batch_trees.ndata['mask'].to(device),
                     wordid=batch_trees.ndata['x'].to(device),
                     label=batch_trees.ndata['y'].to(device),
                     rootid=th.tensor(rootid).to(device))
    return batcher_dev

def main(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    best_epoch = -1
    best_dev_acc = 0

    # only use cpu
    device = th.device('cpu')

    trainset = Dataset(mode=args.train_set)
    train_loader = DataLoader(dataset=trainset,
                              batch_size=args.batch_size,
                              collate_fn=batcher(device),
                              shuffle=True,
                              num_workers=0)
    devset = Dataset(mode=args.dev_set)
    dev_loader = DataLoader(dataset=devset,
                            batch_size=100,
                            collate_fn=batcher(device),
                            shuffle=False,
                            num_workers=0)

    model = TreeLSTM(trainset.vocab_size,
                     args.x_size,
                     args.h_size,
                     trainset.num_classes,
                     args.dropout,
                     cell_type='nary',
                     pretrained_emb = trainset.pretrained_emb).to(device)
    print(model)

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
        training_loss = 0

        t_epoch = time.time()
        model.train()
        for step, batch in enumerate(train_loader):
            g = batch.graph.to(device)
            n = g.number_of_nodes()
            h = th.zeros((n, args.h_size)).to(device)
            c = th.zeros((n, args.h_size)).to(device)
            if step >= 3:
                t0 = time.time()

            logits = model(batch, g, h, c)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, batch.label[batch.rootid], reduction='sum')

            training_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step >= 3:
                dur.append(time.time() - t0)

            if step > 0 and step % args.log_every == 0:
                pred = th.argmax(logits, 1)

                acc = th.sum(th.eq(batch.label[batch.rootid], pred))
                print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Root Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, step, loss.item(), 1.0*acc/len(batch.rootid), np.mean(dur)))
        print('Epoch {:05d} training time {:.4f}s'.format(epoch, time.time() - t_epoch))
        print('Training loss:', training_loss)

        # eval on dev set
        accs = []
        model.eval()
        for step, batch in enumerate(dev_loader):
            g = batch.graph.to(device)
            n = g.number_of_nodes()
            with th.no_grad():
                h = th.zeros((n, args.h_size)).to(device)
                c = th.zeros((n, args.h_size)).to(device)
                logits = model(batch, g, h, c)

            pred = th.argmax(logits, 1)
            acc = th.sum(th.eq(batch.label[batch.rootid], pred))
            accs.append([acc, len(batch.label[batch.rootid])])

        dev_acc = 1.0*np.sum([x[0] for x in accs])/np.sum([x[1] for x in accs])
        print("Epoch {:05d} | Dev Acc {:.4f}".format(epoch, dev_acc))

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch
            th.save(model.state_dict(),
                'checkpoints/best_{}_{}_{}.pkl'.format(args.seed, args.train_set,args.dev_set))
        else:
            if best_epoch <= epoch - 10:
                break

        # lr decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(1e-5, param_group['lr']*0.99) #10
            print(param_group['lr'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--x-size', type=int, default=300)
    parser.add_argument('--h-size', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train-set', type=str, default="train-gh-1")
    parser.add_argument('--dev-set', type=str, default="dev-gh")
    args = parser.parse_args()
    print(args)
    main(args)