from dgl.data import DGLDataset
from collections import OrderedDict
import networkx as nx
import numpy as np
import os
from tqdm import tqdm

import torch as F
from dgl.data.utils import save_graphs, save_info, load_graphs, \
    load_info, deprecate_property
from dgl.convert import from_networkx

class Dataset(DGLDataset):

    PAD_WORD = 0  # special pad word id
    UNK_WORD = 0  # out-of-vocabulary word id

    def __init__(self,
                 mode='train',
                 glove_embed_file="../../data/embedding/glove.840B.300d.txt",
                 vocab_file=None,
                 raw_dir="../../data/",
                 force_reload=False,
                 verbose=False):

        name = "tree"
        self._glove_embed_file = glove_embed_file
        if not os.path.exists(raw_dir+name+"/emb.pkl"):
            self._glove_embed_file = glove_embed_file
        else:
            self._glove_embed_file = None
        self.mode = mode
        self._vocab_file = vocab_file
        super().__init__(name=name,
                         url=None,
                         raw_dir=raw_dir,
                         force_reload=force_reload,
                         verbose=verbose)

    def process(self):
        from nltk.corpus.reader import BracketParseCorpusReader
        # load vocab file
        self._vocab = OrderedDict()
        vocab_file = self._vocab_file if self._vocab_file is not None else os.path.join(self.raw_path, 'vocab.txt')
        with open(vocab_file, encoding='utf-8') as vf:
            for line in vf.readlines():
                line = line.strip()
                self._vocab[line] = len(self._vocab)

        # filter glove
        if self._glove_embed_file is not None and os.path.exists(self._glove_embed_file):
            glove_emb = {}
            with open(self._glove_embed_file, 'r', encoding='utf-8') as pf:
                for line in tqdm(pf.readlines()):
                    sp = line.split(' ')
                    if sp[0].lower() in self._vocab:
                        glove_emb[sp[0].lower()] = np.asarray([float(x) for x in sp[1:]])
        files = ['{}.txt'.format(self.mode)]
        corpus = BracketParseCorpusReader(self.raw_path, files)
        sents = corpus.parsed_sents(files[0])

        # initialize with glove
        pretrained_emb = []
        fail_cnt = 0
        for line in self._vocab.keys():
            if self._glove_embed_file is not None and os.path.exists(self._glove_embed_file):
                if not line.lower() in glove_emb:
                    fail_cnt += 1
                pretrained_emb.append(glove_emb.get(line.lower(), np.random.uniform(-0.05, 0.05, 300)))

        self._pretrained_emb = None
        if self._glove_embed_file is not None and os.path.exists(self._glove_embed_file):
            self._pretrained_emb = F.tensor(np.stack(pretrained_emb, 0))
            print('Miss word in GloVe {0:.4f}'.format(1.0 * fail_cnt / len(self._pretrained_emb)))
        # build trees
        self._trees = []
        for sent in sents:
            self._trees.append(self._build_tree(sent))

    def _build_tree(self, root):
        g = nx.DiGraph()

        def _rec_build(nid, node):
            for child in node:
                cid = g.number_of_nodes()

                # account for trees with a single node
                if isinstance(child, str):
                    return

                if isinstance(child[0], str) or isinstance(child[0], bytes):
                    # leaf node
                    word = self.vocab.get(child[0].lower(), self.UNK_WORD)
                    g.add_node(cid, x=word, y=int(child.label()), mask=1)
                else:
                    g.add_node(cid, x=Dataset.PAD_WORD, y=int(child.label()), mask=0)
                    _rec_build(cid, child)
                g.add_edge(cid, nid)

        # add root
        g.add_node(0, x=Dataset.PAD_WORD, y=int(root.label()), mask=0)
        _rec_build(0, root)
        ret = from_networkx(g, node_attrs=['x', 'y', 'mask'])
        return ret

    def has_cache(self):
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        vocab_path = os.path.join(self.save_path, 'vocab.pkl')
        return os.path.exists(graph_path) and os.path.exists(vocab_path)

    def save(self):
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        save_graphs(graph_path, self._trees)
        vocab_path = os.path.join(self.save_path, 'vocab.pkl')
        save_info(vocab_path, {'vocab': self.vocab})
        if self.pretrained_emb is not None:
            emb_path = os.path.join(self.save_path, 'emb.pkl')
            save_info(emb_path, {'embed': self.pretrained_emb})

    def load(self):
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        vocab_path = os.path.join(self.save_path, 'vocab.pkl')
        emb_path = os.path.join(self.save_path, 'emb.pkl')

        self._trees = load_graphs(graph_path)[0]
        self._vocab = load_info(vocab_path)['vocab']
        self._pretrained_emb = None
        if os.path.exists(emb_path):
            self._pretrained_emb = load_info(emb_path)['embed']

    @property
    def trees(self):
        deprecate_property('dataset.trees', '[dataset[i] for i in len(dataset)]')
        return self._trees

    @property
    def vocab(self):
        return self._vocab

    @property
    def pretrained_emb(self):
        return self._pretrained_emb

    def __getitem__(self, idx):
        return self._trees[idx]


    def __len__(self):
        return len(self._trees)


    @property
    def num_vocabs(self):
        deprecate_property('dataset.num_vocabs', 'dataset.vocab_size')
        return self.vocab_size

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def num_classes(self):
        return 2