import torch as th
import torch.nn.functional as F
import numpy as np
import stanza
from collections import OrderedDict
from tqdm import tqdm
import pickle
from os.path import exists
import collections

def tokenize(text, nlp):
    doc = nlp(text)
    result = []
    for sentence in doc.sentences:
        result.append([token.text for token in sentence.tokens])
    return result

Batch = collections.namedtuple('Batch', ['wordid', 'lengths','label'])

def batcher(data):
    sents, labels = zip(*data)
    max_len = max(map(len, sents))
    seq_tensor = []
    sl = []
    for sent in sents:
        seq_tensor.append(th.cat([th.Tensor(sent), th.zeros(max_len-len(sent))]))
        sl.append(len(sent))
    seq_tensor = th.Tensor(np.stack(seq_tensor, 0))
    seq_lengths = th.Tensor(sl)
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    labels = th.Tensor(labels)[perm_idx]

    return Batch(seq_tensor.long(), seq_lengths, labels.long())

class Dataset(th.utils.data.Dataset):
    
    def __init__(self, mode):
        self.PAD_WORD = -1  # special pad word id
        self.UNK_WORD = 0  # out-of-vocabulary word id
        # load vocab file
        self._vocab = OrderedDict()
        with open("../../data/label/vocab.txt", encoding='utf-8') as vf:
            for line in vf.readlines():
                line = line.strip()
                self._vocab[line] = len(self._vocab)
        
        self.vocab_size = len(self._vocab)
        
        if exists('../../data/label/emb.pkl'):
            with open('../../data/label/emb.pkl', 'rb') as handle:
                self.pretrained_emb = pickle.load(handle)
        else:
            glove_emb = {}
            with open("../../data/embedding/glove.840B.300d.txt", 'r', encoding='utf-8') as pf:
                for line in tqdm(pf.readlines()):
                    sp = line.split(' ')
                    if sp[0].lower() in self._vocab:
                        glove_emb[sp[0].lower()] = np.asarray([float(x) for x in sp[1:]])

            # initialize with glove
            pretrained_emb = []
            for line in self._vocab.keys():
                pretrained_emb.append(glove_emb.get(line.lower(), np.random.uniform(-0.05, 0.05, 300)))

            self.pretrained_emb = F.Tensor(np.stack(pretrained_emb, 0))
            
            with open('../../data/label/emb.pkl', 'wb') as handle:
                pickle.dump(self.pretrained_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        stanza.download('en')
        nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)

        self.sents = []
        self.labels = []
        with open("../../data/label/"+mode+".txt") as f:

            # parse all sentences at once to speed up
            all_sentences = ""
            for line in f:
                all_sentences += line[:-3] + '\n\n'
                self.labels.append(int(line[-2]))
            
            for sent in tokenize(all_sentences, nlp):
                self.sents.append([self._vocab.get(word, self.UNK_WORD) for word in sent])

            print("Finished tokenization for {}!".format(mode))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sents[index], self.labels[index]