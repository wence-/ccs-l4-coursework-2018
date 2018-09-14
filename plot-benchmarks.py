#!/usr/bin/env python
from matplotlib import pyplot
import numpy

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(description="Plot benchmark results",
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--data-file", help="Data file to read",
                    default="benchmark.dat")

parser.add_argument("--output-file", help="Output file (PDF) to write plot to",
                    default="benchmark.pdf")

args = parser.parse_args()

try:
    data = numpy.loadtxt(args.data_file)
except OSError:
    raise ValueError("Unable to load benchmark data from file %s" % args.data_file)

_, cols = data.shape
if cols != 5:
    raise ValueError("Expected 5 columns in benchmark data, found %d" % cols)

for row in data:
    try:
        m, = numpy.unique(row[:3])
    except ValueError:
        raise ValueError("This plotting script only knows how to handle square matrices\n"
                         "You have %s" % tuple(row[:3]))

fig = pyplot.figure(figsize=(9, 5), frameon=False)
ax = fig.add_subplot(111)
ax.set_xlabel("Matrix size")
ax.set_ylabel("GFLOP/s")

size = data[:, 0]
time = data[:, 3]
flop = data[:, 4]
gflops = (flop/time) * 1e-9

ax.set_ylim(0, gflops.max())

ax.plot(size, gflops, linewidth=2, linestyle="solid")

fig.savefig(args.output_file,
            orientation="landscape",
            format="pdf",
            transparent=True)
