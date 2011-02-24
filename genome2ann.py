#!/usr/bin/python3.1

from optparse import OptionParser
from environment import Brain
from main import DESCRIPTOR
from sys import stdin
from age import Genome

def load_chromosomes(f):
    return list(filter(lambda l: l!="" and not l.startswith("#"), map(lambda l: l.strip(), f.readlines())))

# parse options
parser = OptionParser(usage = "Usage: %prog [OPTIONS] [FILE]")
parser.add_option("-o", "--output", dest="output", default="net.pcn",
                  help="Write ANN to FILE", metavar="FILE")
parser.add_option("-x", "--export", dest="export", default=None,
                  help="Export as FORMAT", metavar="FORMAT")
(options, args) = parser.parse_args()

# load chromosomes
if (len(args)<1):
    load_chromosomes(stdin)
else:
    chromosomes = load_chromosomes(open(args[0], "rt"))
    
# create genome
genome = Genome(chromosomes = chromosomes, desc = DESCRIPTOR)
genome.parse()

# create network
net = Brain(genome)

# save network
net.save(options.output, options.export)
