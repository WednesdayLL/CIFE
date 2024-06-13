import time
from subprocess import check_output
import numpy as np


def main():
    cancers = ['PANCAN']
    option = ['embedding','train', 'test','casual']

    for c in cancers:
        for op in option:
            cmd = 'python main_pytorch.py --mode %s --type %s' % (op, c)
            print(cmd)
            o = check_output(cmd, shell=True, universal_newlines=True)
            print(o)

if __name__ == "__main__":
    main()
