# PETCI: A Parallel English Translation Dataset of Chinese Idioms

PETCI is a **P**arallel **E**nglish **T**ranslation dataset of **C**hinese **I**dioms, collected from an idiom dictionary and Google and DeepL translation. PETCI contains **4,310** Chinese idioms with **29,936** English translations. These translations capture diverse translation errors and paraphrase strategies.

We provide several baseline models to facilitate future research on this dataset.

## Data

The Chinese idioms and their translations are in the `./data/json/raw.json` file. Here is one example:

```python
{
    "id": 0,
    "chinese": "一波未平，一波又起",
    "book": [
        "suffer a string of reverses",
        "hardly has one wave subsided when another rises",
        "one trouble follows another"
    ],
    "google": [
        "One wave is not flat, another wave is rising"
    ],
    "deepl": [
        "before the first wave subsides, a new wave rises"
    ]
}
```

- `id` is the index of the idiom in the dictionary
- `chinese` is the Chinese idiom
- `book` is the translations from the dictionary
- `google` is the translation from Google
- `deepl` is the translation from DeepL

In `./data/json/filtered.json`, the `machine` translations that are the same as dictionary translations are removed, and the dictionary translations are split into `gold` and `human` translations.

## Training and Testing

### Prerequisites

Run `pip install -r ./models/requirements.txt` to install required packages. Download and put `glove.840B.300d.txt` in `./data/embedding`. Download [CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html).

### Create Datasets
Before training, run `java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -parse.binaryTrees` to start the CoreNLP server, and run the following commands in the `./data` folder to create the necessary datasets.

```shell
mkdir label simplify tree
python dataset.py
```

### LSTM

In the enclosing folder, run
```shell
./auto_train.sh
./auto_test.sh
```

### Tree-LSTM

In the enclosing folder, run
```shell
./auto_train.sh
./auto_test.sh
```

### BERT

In the enclosing folder, run
```shell
SEED=45
HM=ghm
PART=5
python train.py --seed $SEED --train-set train-$HM-$PART --dev-set dev-$HM

MODEL=checkpoint-5000
python test.py --model $MODEL --test-set dev-$HM --seed $SEED --hm $HM --part $PART
```

### NTS

In the enclosing folder, run
```shell
onmt_build_vocab -config vocab.yaml -n_sample -1 

onmt_train -config nts.yaml

BEST=checkpoints/checkpoint_step_300.pt
SRC=../../data/simplify/test-src.txt
OUTPUT=../test-output.txt
onmt_translate -model $BEST -src $SRC -output $OUTPUT -verbose -beam_size 5
```

### Figures
In the `figs` folder, run `python plot.py --model lstm`, where the model name can be replaced by `tree_lstm` or `bert`.
