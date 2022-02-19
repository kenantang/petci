import argparse
from os.path import exists
import json
import numpy as np
import matplotlib.pyplot as plt

labels = ["gh", "gm", "ghm"]

def main(args):
    with open(args.size_file, "r") as f:
        size = json.load(f)

    accs = {}
    tests = []

    with open(args.summary, "r") as f:
        for line in f:
            result = json.loads(line)
            info = result['model'].split('_')
            train = info[2]
            test = result['test-set']
            if test not in tests:
                tests.append(test)
            acc = result['accuracy']

            if train not in accs:
                accs[train] = {}
            if test in accs[train]:
                accs[train][test].append(acc)
            else:
                accs[train][test] = [acc]
            

    # 4 plots
    # g acc, h acc, m acc, total acc

    figs = {}
    axs = {}
    for t in tests:
        figs[t], axs[t] = plt.subplots()
        axs[t].set_ylim(0, 1)

        accs_mean = {}
        accs_sd = {}
        accs_sizes = {}

        for l in labels:
            accs_mean[l] = []
            accs_sd[l] = []
            accs_sizes[l] = []

        for k in size:
            for l in labels:
                # plot 3 lines, gh-, gm-, ghm-
                if l+'-' in k:
                    accs_sizes[l].append(size[k])
                    accs_mean[l].append(np.mean(accs[k][t]))
                    accs_sd[l].append(np.std(accs[k][t]))

        print(t)

        for l in labels:
            with open("accs/{}-{}-{}.dat".format(args.model, t, l), "w") as fd:
                for idx in range(len(accs_sizes[l])):
                    fd.write("{}\t{:.2f}\t{:.2f}\n".format(
                        accs_sizes[l][idx],
                        accs_mean[l][idx]*100,
                        accs_sd[l][idx]*100
                    ))

                    # print the result in latex table format
                    print(" & {:.2f} ({:.2f})".format(accs_mean[l][idx]*100, accs_sd[l][idx]*100), end="")
            print()
        
        for l in labels:
            axs[t].errorbar(accs_sizes[l], accs_mean[l], yerr=accs_sd[l], label=l)


    for t in tests:
        axs[t].legend()
        figs[t].savefig('{}-{}.png'.format(args.model, t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size-file', type=str, default="../data/json/size.json")
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--summary', type=str, default=None)
    args = parser.parse_args()

    assert args.model is not None, "Model not specified!"
    args.summary = "../models/"+args.model+"/test_summary.jsonl"
    assert exists(args.summary), "Summary file does not exist!"
    main(args)