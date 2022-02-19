import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 num_classes,
                 dropout,
                 pretrained_emb=None):
        super(LSTM, self).__init__()
        self.x_size = x_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            print('Using glove')
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(x_size, h_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)

    def forward(self, batch):
        # no initial h and c, set to zero by default
        embeds = self.embedding(batch.wordid)
        embeds = self.dropout(embeds)
        packed = pack_padded_sequence(embeds, batch.lengths, batch_first=True)
        _, (h, _) = self.lstm(packed)
        h = self.dropout(h)
        logits = self.linear(h[-1])
        return logits