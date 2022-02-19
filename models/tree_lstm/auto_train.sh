#!/bin/bash

# use 5 seeds
for SEED in 41 42 43 44 45
do
    # use different training sets
    for HM in gh gm ghm
    do
        # use different training set sizes
        for PART in 1 2 3 4 5
        do
            python -u train.py --seed $SEED --train-set train-$HM-$PART --dev-set dev-$HM > log/$SEED-$HM-$PART.txt
        done
    done
done
