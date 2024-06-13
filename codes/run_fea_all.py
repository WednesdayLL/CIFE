import time
from subprocess import check_output
import numpy as np


def main():
    cancers = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC',
               'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP',
               'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD',
               'PANCAN', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM',
               'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
    cancers = ['PANCAN']
    option = ['train', 'test','casual']
    # option = ['test','casual']
    # option = ['test']
    # option = [ 'embedding', 'train', 'test']
    # option = ['embedding']
    hidden_size1 = [30, 28, 26, 24, 22]  # 54,50,   40,30,22,18
    hidden_size2 = [10, 8, 6]
    # # hidden_size3 = [12,10,8,6]
    # learning_rate = [0.0014,0.0015,0.0016,0.0017] # 0.001,0.0012,0.0013,0.0014
    learning_rate = [
        # 0.00141,0.00142,0.00143,0.00144,0.00145,0.00146,0.00147,
        0.00148, 0.00149, 0.0015, 0.00151, 0.00152, 0.00153, 0.00154,
        # 0.00155,0.00156,0.00157,0.00158,0.00159
    ]  # 0.001,0.0012,0.0013,0.0014
    # batch_size = [ 8]
    dmodel = [34, 32, 30]
    # dmodel = [40, 38, 36, ]
    nhead = [2, 3, 4, 5, 6]
    dimFF = [70, 68, 66, 64, 62, 60]

    # scarf`
    # embedding
    # embeddings = [50,47,45,]
    # embeddings = [45,43,41,39,37,]
    embeddings = [44, 42, 40, 38, 36]

    # corruption_rate=[0.5,0.6,0.7]
    # encoder_depth = [1,2,3,4,5]
    # head_depth = [1,2,3]

    corruption_rate = [0.6, ]
    encoder_depth = [4, ]
    head_depth = [2]

    # seeds = [i for i in range(201,601)]
    seeds = [i for i in range(601, 1001)]
    # print(seeds)

    # for s in seeds:
    #     for op in option:
    #         cmd = 'python main_pytorch.py --mode %s --seed %d' % (op,s)
    #         print(cmd)
    #         o = check_output(cmd, shell=True, universal_newlines=True)
    #         print(o)
    for c in cancers:
        for op in option:
            cmd = 'python main_pytorch.py --mode %s --type %s' % (op, c)
            print(cmd)
            o = check_output(cmd, shell=True, universal_newlines=True)
            print(o)
    # for lr in learning_rate:
    #     for op in option:
    #         cmd = 'python main_pytorch.py --mode %s --lr %f' % (op,lr)
    #         print(cmd)
    #         o = check_output(cmd, shell=True, universal_newlines=True)
    #         print(o)

    # for cr in corruption_rate:
    #         for ed in encoder_depth:
    #             for hd in head_depth:
    #                 for embed in embeddings:
    #                     cmd = 'python main_pytorch.py --mode embedding --emb_dim %d --head_depth %d --corruption_rate %f --encoder_depth %d' % (embed, hd, cr, ed)
    #                     print(cmd)
    #                     o = check_output(cmd, shell=True, universal_newlines=True)
    #                     print(o)
    #                     for hz1 in hidden_size1:
    #                         for hz2 in hidden_size2:
    #                             for lr in learning_rate:
    #                                 for op in option:
    #                                     cmd = 'python main_pytorch.py --mode %s --lr %f --hz1 %d --hz2 %d --emb_dim %d --head_depth %d --corruption_rate %f --encoder_depth %d' % (op, lr, hz1, hz2, embed, hd, cr, ed)
    #                                     print(cmd)
    #                                     o = check_output(cmd, shell=True, universal_newlines=True)
    #                                     print(o)


if __name__ == "__main__":
    main()
