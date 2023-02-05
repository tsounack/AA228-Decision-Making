import sys

import graph


def compute(infile, outfile):
    G = graph.Graph(infile)
    G.K2_search()
    G.export_bayesian_score(outfile)
    G.draw_gph(outfile)
    G.write_gph(outfile)


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
