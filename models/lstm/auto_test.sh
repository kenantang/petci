#!/bin/bash

DIRECTORY=batch-25
echo $DIRECTORY

# WARNING: remove the test results from previous runs
rm test_summary.jsonl

# test all 75 models
for SEED in 41 42 43 44 45
do
    for HM in gh gm ghm
    do
        for PART in 1 2 3 4 5
        do
            MODEL=best_$SEED\_train-$HM-$PART\_dev-$HM.pkl

            # test on all four test sets
            for SET in g h m all
            do
                python test.py --directory $DIRECTORY --model $MODEL --test-set test-$SET
            done
        done
    done
done