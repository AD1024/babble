import subprocess
from dataclasses import dataclass
import sys
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.pyplot import cm
import numpy as np

# NOTE: For max memory, we need a different workflow that
# can't use the exisiting experiment infrastructure.
# We have to run each configuration separately and
# log the max memory consumption.
def param_sweep_old(path_to_drawing_bab, single_run_data, alldata):
    fw = open(alldata, "w")
    allwriter = csv.writer(fw)
    # beams = [10, 50, 100, 200, 500, 1000]
    # lps = [1, 3, 5, 10]
    # rounds = [2, 5, 10]
    beams = [10, 25, 50, 100]
    lpss = [1, 2, 5, 10, 25]
    rounds = [100]
    max_arity = 3
    for b in beams:
        for lps in lpss:
            for round in rounds:
                bm = str(b).split()[0]
                lp = str(lps).split()[0]
                rn = str(round).split()[0]
                _, e = subprocess.Popen(["gtimeout", "-v", "100s", "/usr/bin/time", "-l", "cargo", "run", "--release", "--bin=drawings", "--",
                               path_to_drawing_bab, "--beams", bm, "--lps", lp, "--rounds", rn, "--max-arity", str(max_arity)],
                               stderr=subprocess.PIPE).communicate()
                mem = ""
                if "TERM" in (str(e)):
                    print("CONFIG beam: {0}, lps: {1}, round: {2} TIMED OUT".format(bm, lp, rn))
                    continue
                ls = str(e).split()
                if "maximum" in ls:
                    max_idx = ls.index("maximum")
                    if "resident" in ls and (ls.index("resident") == max_idx + 1):
                        max_mem = ls[max_idx - 1]
                        mem = str(max_mem)
                    else:
                        mem = ""
                with open(single_run_data, 'r') as fr:
                    single_reader = csv.reader(fr)
                    for row in single_reader:
                        row.append(mem)
                        allwriter.writerow(row)
    fw.close()


def param_sweep(path_to_drawing_bab):
    # beams = "10 50 100 200 500 1000"
    # lps = "1 3 5 10"
    # rounds = "2 5 10"
    beams = [10]
    lpss = [1]
    rounds = [1]
    max_arity = 3
    for b in beams:
        for lps in lpss:
            for round in rounds:
                bm = str(b).split()[0]
                lp = str(lps).split()[0]
                rn = str(round).split()[0]
                _, e = subprocess.Popen(["gtimeout", "-v", "100s", "/usr/bin/time", "-l", "cargo", "run", "--release", "--bin=drawings", "--",
                               path_to_drawing_bab, "--beams", bm, "--lps", lp, "--rounds", rn, "--max-arity", str(max_arity)],
                               stderr=subprocess.PIPE).communicate()
                if "TERM" in (str(e)):
                    print("CONFIG beam: {0}, lps: {1}, round: {2} TIMED OUT".format(bm, lp, rn))
                    continue

def parse_results_csv(path):
    FIELDS = \
        ['exp_type', 'timeout', 'beam_size', 'beam_size_2', 'lps', 'extra_por',
            'extra_data', 'round', 'init_size', 'final_size', 'compression', 'time', 'memory']
    with open(path) as f:
        rows = list(csv.DictReader(f, fieldnames=FIELDS))

    return rows

# TODO: Incorporate memory into this somehow
def mk_cactus(rows, plot_dir):
    # We're makin a lil cactus plot type thing!
    # The idea is that we have a line corresponding to a pairing of
    # beam size and lps, where the data points on the line correspond
    # to the number of rounds we run (and thus the time used).
    # We only want to add a point to the line if the extra time to run
    # more rounds results in higher compression.
    group = {}

    for r in rows:
        g = (r["beam_size"], r["lps"])
        if g not in group:
            group[g] = []

        insert = True

        # for og in group[g]:
        #     # Out of the entries already in this group, if they take less time,
        #     # they must have a lower compression ratio. If this isn't the case,
        #     # we don't add the row.
        #     if og['time'] < r['time'] and og['compression'] > r['compression']:
        #         insert = False

        if insert:
            group[g].append(r)

    plt.figure()
    color = iter(cm.rainbow(np.linspace(0, 1, len(group))))
    for g in group:
        xs = [float(r['time']) for r in group[g]]
        ys = [float(r['compression']) for r in group[g]]
        plt.plot(xs, ys, marker="o", c=next(color), label=str(g))
    fnm = os.path.join(plot_dir, 'cactus-time-compression.pdf')
    plt.legend(loc="lower right")
    plt.title('time v. compression over all (beam size, lps)')
    plt.savefig(fnm)


def mkplot(rows, xField, yField, plot_dir):
    assert xField in ['round', 'beam_size', 'lps']
    assert yField in ['compression', 'time', 'memory']

    groupBy = ['round', 'beam_size', 'lps']
    groupBy.remove(xField)

    group = {}
    for r in rows:
        g = tuple([r[f] for f in groupBy])
        if g not in group:
            group[g] = []
        group[g].append(r)

    plt.figure()
    color = iter(cm.rainbow(np.linspace(0, 1, len(group))))
    for g in group:
        xs = [float(r[xField]) for r in group[g]]
        ys = [float(r[yField]) for r in group[g]]
        plt.plot(xs, ys, marker="o", c=next(color), label=str(g))
    fnm = os.path.join(plot_dir, '{}-{}.pdf'.format(xField, yField))
    plt.legend(loc="upper right")
    plt.title('{} vs {} over all {}'.format(yField, xField, ','.join(groupBy)))
    plt.savefig(fnm)


def mkplots(rows):
    plot_dir = os.path.join('plot_results')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    mk_cactus(rows, plot_dir)

    # for x in ['round', 'beam_size', 'lps']:
    #     for y in ['compression', 'time', 'memory']:
    #         mkplot(rows, x, y, plot_dir)

def analyze_data(p):
    mkplots(parse_results_csv(p))


usage = """USAGE: python param_sweep.py path_to_drawing_benchmark.bab
"""

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print(usage)
    else:
        fnm = sys.argv[1]
        if (fnm.split(".")[1] != "bab"):
            print("Must provide .bab file")
        else:
            param_sweep(sys.argv[1])
            #param_sweep(sys.argv[1], "target/res_drawing.csv", "target/alldata.csv")
            analyze_data("target/res_drawing.csv")
            #analyze_data("target/alldata.csv")
