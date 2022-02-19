import argparse
import stanza
import json
import numpy as np
from os.path import exists
import requests
import json
import re
from tqdm import tqdm

# clean half nodes from a binary parse
def clean_binary_parse(bp):
    bp = bp.replace('\n', '')
    bp = re.sub(' +', ' ', bp)
    bp = re.sub('\((.*?) ', '(0 ', bp)
    
    # construct a tree from the string
    
    words = []
    words_lc = []
    words_rc = []
    words_p = []
    word = ""
    root_p = -1
    cur_p = -1
    for c in bp:
        if c != '(' and c != ')':
            word += c
        else:
            if not word.isspace() and word!='':
                words.append(word)
                words_lc.append(None)
                words_rc.append(None)
                words_p.append(cur_p)
                cur_c = len(words_p)-1
                if cur_p != root_p:
                    if not words_lc[cur_p]:
                        words_lc[cur_p] = cur_c
                    else:
                        words_rc[cur_p] = cur_c
                cur_p = cur_c
            if c == ')':
                cur_p = words_p[cur_p]
            word = ""
    
    # remove half-nodes
    removed = [False for _ in words]
    for i in range(len(words)):
        
        # by our construction, any half-node always has left child
        if words_lc[i] and not words_rc[i]:
            g = words_p[i]
            c = words_lc[i]
            words_p[c] = g
            
            # we do not want returns the last element for index==-1
            if g != root_p:
                if words_lc[g] == i:
                    words_lc[g] = c
                else:
                    words_rc[g] = c

            removed[i] = True
        if words_p[i] == root_p and not removed[i]:
            root = i
            
    def inorder(root):
        if not root:
            return ''
        s = '(' + words[root]
        s += inorder(words_lc[root])
        if words_lc[root]:
            s += ' '
        s += inorder(words_rc[root]) + ')'
        return s

    # in the end, there should be 2n-1 labels if there are n tokens
    return inorder(root)

def main(args):
    np.random.seed(args.seed)

    # open the json
    with open("json/filtered.json", "r") as f:
        filtered = json.load(f)

    all_sentences = ""
    for i in filtered:
        all_sentences += i["gold"].lower() + '\n\n'
        for h in i["human"]:
            all_sentences += h.lower() + '\n\n'
        for m in i["machine"]:
            all_sentences += m.lower() + '\n\n'

    # create vocabulary for the labelled sentences
    # required by LSTM but not by BERT
    if exists("label/vocab.txt"):
        print("Vocabulary for labelled sentences exists!")
    else:
        stanza.download('en')

        vocab = set()
        nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)

        print("Creating vocabulary for labelled sentences...")

        doc = nlp(all_sentences)
        for sentence in doc.sentences:
            for token in sentence.tokens:
                vocab.add(token.text)
        
        with open("label/vocab.txt", "w") as fo:
            fo.write("<UNK>\n")
            for v in list(vocab):
                fo.write(v+"\n")

    
    # get parse results of all sentences
    parsed_sentences = []

    if exists("tree/parse.txt"):
        # need to manually check if this file is complete
        with open("tree/parse.txt", "r") as f:
            for line in f:
                parsed_sentences.append(json.loads(line))
    else:
        # parse sentences from scratch, may take around 10 minutes!
        r = 'http://[::]:9000/?properties={"annotators":"tokenize,ssplit,pos,parse","outputFormat":"json"}'
        separate_sentences = all_sentences.split('\n\n') # last one is empty

        # file to save the parsed sentence
        fp = open("tree/parse.txt", "w")

        for s in tqdm(separate_sentences):
            j = json.loads(requests.post(r, data = s).text)

            # save only relevant fields
            j_rel = {}

            # clean the binary parse right after results are returned
            j_rel['binaryParse'] = clean_binary_parse(j['sentences'][0]['binaryParse'])
            
            # save tokens for building vocabulary
            j_rel['tokens'] = j['sentences'][0]['tokens']
            parsed_sentences.append(j_rel)
            json.dump(parsed_sentences[-1], fp)
            fp.write('\n')

    # read the parsed sentence for each idiom
    parsed_idiom = []
    idx = 0
    for i in filtered:
        parsed_idiom.append({"gold": parsed_sentences[idx]})

        # change to gold label on root node!
        parsed_idiom[-1]["gold"]["binaryParse"] = '(1' + parsed_idiom[-1]["gold"]["binaryParse"][2:]
        idx += 1
        parsed_idiom[-1]["human"] = []
        parsed_idiom[-1]["machine"] = []
        for h in i["human"]:
            parsed_idiom[-1]["human"].append(parsed_sentences[idx])
            idx += 1
        for m in i["machine"]:
            parsed_idiom[-1]["machine"].append(parsed_sentences[idx])
            idx += 1

    # create vocabulary for the constituency parse trees
    # start the server by:
    # java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -parse.binaryTrees
    if exists("tree/vocab.txt"):
        print("Vocabulary for parsed sentences exists!")
    else:
        vocab = set()

        print("Creating vocabulary for parsed sentences...")

        for sentence in parsed_sentences:
            for token in sentence['tokens']:
                vocab.add(token['word'])
        
        with open("tree/vocab.txt", "w") as fo:
            fo.write("<UNK>\n")
            for v in list(vocab):
                fo.write(v+"\n")

    assert args.points >= 2, "Too few data points!"

    idiom_total = len(filtered)
    split = {"train": [], "dev": [], "test": []}
    idx_train = int(idiom_total * 0.8)
    idx_dev = int(idiom_total * 0.9)
    all_idx =list(range(idiom_total))

    # need to shuffle, because idioms of similar meaning are put together in the dictionary
    np.random.shuffle(all_idx)
    split['train'] = all_idx[0:idx_train]
    split['dev'] = all_idx[idx_train:idx_dev]
    split['test'] = all_idx[idx_dev:idiom_total]
    print("train/dev/test: {}, {}, {}".format(idx_train, idx_dev-idx_train, idiom_total-idx_dev))

    with open("json/split.json", "w") as fo:
        json.dump(split, fo)

    # all lower case
    for w in filtered:
        w["gold"] = w["gold"].lower()
        w["human"] = [h.lower() for h in w["human"]]
        w["machine"] = [m.lower() for m in w["machine"]]

    # train set

    hum_head = []
    mac_head = []
    hum_tail = []
    mac_tail = []

    filtered_train = [filtered[i] for i in split["train"]]
    parsed_idiom_train = [parsed_idiom[i] for i in split["train"]]

    for idx, i in enumerate(filtered_train):
        lh = len(i["human"])
        if lh > 0:
            hum_head.append((idx, 0))
        if lh > 1:
            hum_tail.extend([(idx, j) for j in range(1, lh)])
        mh = len(i["machine"])
        if mh > 0:
            mac_head.append((idx, 0))
        if mh > 1:
            mac_tail.extend([(idx, j) for j in range(1, mh)])

    np.random.shuffle(hum_tail)
    np.random.shuffle(mac_tail)
    hum_all = hum_head + hum_tail
    mac_all = mac_head + mac_tail

    hum_indices = np.linspace(len(hum_head), len(hum_all), args.points, dtype=int)
    mac_indices = np.linspace(len(mac_head), len(mac_all), args.points, dtype=int)

    print("Number of human translations for training:", hum_indices)
    print("Number of machine translations for training:", mac_indices)

    sizes = {}
    for i in range(args.points):
        name = str(i+1)
        sizes["train-gh-"+name] = int(hum_indices[i])
        sizes["train-gm-"+name] = int(mac_indices[i])
        sizes["train-ghm-"+name] = int(hum_indices[i]+mac_indices[i])

    # save training set size for plotting
    with open("json/size.json", "w") as f:
        json.dump(sizes, f, indent=4)

    for i, (hr, mr) in enumerate(zip(hum_indices, mac_indices)):
        hl = 0
        ml = 0

        file_idx = str(i+1)

        # save labelled sentences
        foh_train = open("label/train-gh-"+file_idx+".txt", "w")
        fom_train = open("label/train-gm-"+file_idx+".txt", "w")
        fohm_train = open("label/train-ghm-"+file_idx+".txt", "w")

        # save parsed sentences
        th_train = open("tree/train-gh-"+file_idx+".txt", "w")
        tm_train = open("tree/train-gm-"+file_idx+".txt", "w")
        thm_train = open("tree/train-ghm-"+file_idx+".txt", "w")

        # save simplification
        if i == args.points - 1:
            ss_train = open("simplify/train-src.txt", "w")
            st_train = open("simplify/train-tgt.txt", "w")
        
        # balance training data

        for j in range(hl, hr):
            i_idx, j_idx = hum_all[j]

            # labelled
            foh_train.write(filtered_train[i_idx]["gold"] + " 1\n")
            foh_train.write(filtered_train[i_idx]["human"][j_idx] + " 0\n")
            fohm_train.write(filtered_train[i_idx]["gold"] + " 1\n")
            fohm_train.write(filtered_train[i_idx]["human"][j_idx] + " 0\n")

            # tree
            th_train.write(parsed_idiom_train[i_idx]["gold"]['binaryParse']+"\n")
            th_train.write(parsed_idiom_train[i_idx]["human"][j_idx]['binaryParse']+"\n")
            thm_train.write(parsed_idiom_train[i_idx]["gold"]['binaryParse']+"\n")
            thm_train.write(parsed_idiom_train[i_idx]["human"][j_idx]['binaryParse']+"\n")

            # simplify, do not save parts, but all
            if i == args.points - 1:
                ss_train.write(filtered_train[i_idx]["human"][j_idx]+'\n')
                st_train.write(filtered_train[i_idx]["gold"]+'\n')

        for j in range(ml, mr):
            i_idx, j_idx = mac_all[j]

            # labelled
            fom_train.write(filtered_train[i_idx]["gold"] + " 1\n")
            fom_train.write(filtered_train[i_idx]["machine"][j_idx] + " 0\n")
            fohm_train.write(filtered_train[i_idx]["gold"] + " 1\n")
            fohm_train.write(filtered_train[i_idx]["machine"][j_idx] + " 0\n") 

            # tree
            tm_train.write(parsed_idiom_train[i_idx]["gold"]['binaryParse']+"\n")
            tm_train.write(parsed_idiom_train[i_idx]["machine"][j_idx]['binaryParse']+"\n")
            thm_train.write(parsed_idiom_train[i_idx]["gold"]['binaryParse']+"\n")
            thm_train.write(parsed_idiom_train[i_idx]["machine"][j_idx]['binaryParse']+"\n")

            # simplify
            if i == args.points - 1:
                ss_train.write(filtered_train[i_idx]["machine"][j_idx]+'\n')
                st_train.write(filtered_train[i_idx]["gold"]+'\n')

    # dev set

    foh_dev = open("label/dev-gh.txt", "w")
    fom_dev = open("label/dev-gm.txt", "w")
    fohm_dev = open("label/dev-ghm.txt", "w")

    th_dev = open("tree/dev-gh.txt", "w")
    tm_dev = open("tree/dev-gm.txt", "w")
    thm_dev = open("tree/dev-ghm.txt", "w")

    ss_dev = open("simplify/dev-src.txt", "w")
    st_dev = open("simplify/dev-tgt.txt", "w")

    for i in split["dev"]:
        w = filtered[i]
        wp = parsed_idiom[i]
        
        foh_dev.write(w["gold"] + " 1\n")
        fom_dev.write(w["gold"] + " 1\n")

        th_dev.write(wp["gold"]['binaryParse']+'\n')
        tm_dev.write(wp["gold"]['binaryParse']+'\n')

        # labelled and simplify
        for h in w["human"]:
            foh_dev.write(h + " 0\n")
            fohm_dev.write(h + " 0\n")

            ss_dev.write(h+'\n')
            st_dev.write(w["gold"]+'\n')

        for m in w["machine"]:
            fom_dev.write(m + " 0\n")
            fohm_dev.write(m + " 0\n")

            ss_dev.write(m+'\n')
            st_dev.write(w["gold"]+'\n')

        # tree
        for h in wp["human"]:
            th_dev.write(h['binaryParse']+'\n')
            thm_dev.write(h['binaryParse']+'\n')
        for m in wp["machine"]:
            tm_dev.write(m['binaryParse']+'\n')
            thm_dev.write(m['binaryParse']+'\n')


    # test set

    fog_test = open("label/test-g.txt", "w")
    foh_test = open("label/test-h.txt", "w")
    fom_test = open("label/test-m.txt", "w")
    foa_test = open("label/test-all.txt", "w")

    tg_test = open("tree/test-g.txt", "w")
    th_test = open("tree/test-h.txt", "w")
    tm_test = open("tree/test-m.txt", "w")
    ta_test = open("tree/test-all.txt", "w")

    ss_test = open("simplify/test-src.txt", "w")
    st_test = open("simplify/test-tgt.txt", "w")


    for i in split["test"]:
        # label
        w = filtered[i]
        fog_test.write(w["gold"] + " 1\n")
        foa_test.write(w["gold"] + " 1\n")
        for h in w["human"]:
            foh_test.write(h + " 0\n")
            foa_test.write(h + " 0\n")

            ss_test.write(h+'\n')
            st_test.write(w["gold"]+'\n')

        for m in w["machine"]:
            fom_test.write(m + " 0\n")
            foa_test.write(m + " 0\n")

            ss_test.write(m+'\n')
            st_test.write(w["gold"]+'\n')

        # tree
        wp = parsed_idiom[i]
        tg_test.write(wp["gold"]["binaryParse"]+'\n')
        ta_test.write(wp["gold"]["binaryParse"]+'\n')
        for h in wp["human"]:
            th_test.write(h["binaryParse"]+'\n')
            ta_test.write(h["binaryParse"]+'\n')
        for m in wp["machine"]:
            tm_test.write(m["binaryParse"]+'\n')
            ta_test.write(m["binaryParse"]+'\n')
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('-p', '--points', type=int, default=5)
    args = parser.parse_args()
    main(args)