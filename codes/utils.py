import math

import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu


def fishertest(score_list, key):
    a = 0
    for s in score_list:
        if s in key:
            a += 1
    b = len(score_list) - a
    c = len(key) - a

    d = 20000 - len(score_list) - b
    table = np.array([[a, c], [b, d]])
    # table = np.array([a, b, c, d])

    # table = np.array([[a, b], [c, d]])
    oddsratio, pvalue = fisher_exact(table, alternative='greater')
    # p = pvalue
    p = -math.log10(pvalue)
    # return a
    # print(a,b,c,d,pvalue)
    # print(a/len(key),a/len(score_list))
    return p

# 约登指数寻找最优阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    # print('TPR',TPR)
    # print('FPR', TPR)
    # print('threshold',threshold)
    y = TPR - FPR
    # print('y',y)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    # print('Youden_index',Youden_index)
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def fisher_ex(a, b, c, d):
    _, pvalue = fisher_exact([[a, b], [c, d]], 'greater')
    if pvalue < 1e-256:
        pvalue = 1e-256
    p1 = -math.log10(pvalue)
    # p1 = pvalue
    return p1

def fisher(gene_sig, gene_nsig, key):
    gene_sig_true = [item for item in gene_sig if item in key]
    gene_nsig_true = [item for item in gene_nsig if item in key]
    sig_true_len = len(gene_sig_true)
    sig_flase_len = len(gene_sig) - len(gene_sig_true)
    nsig_true_len = len(gene_nsig_true)
    number_cancer_1 = 20000 - len(gene_sig) - nsig_true_len
    p = fisher_ex(sig_true_len, sig_flase_len, nsig_true_len, number_cancer_1)
    # len(gene_sig_true) / len(key), len(gene_sig_true) / len(gene_sig)
    return p

# use the Mann-Whitney U test to calculate the p-value
def mannwhitneyu_(probas_, yy):
    yy = np.array(yy).flatten()
    x1_all = []
    x2_all = []
    for i in probas_[yy == 0]:
        x1_all.append(i)
    for i in probas_[yy == 1]:
        x2_all.append(i)
    statistic, pvalue = mannwhitneyu(x1_all, x2_all, use_continuity=True, alternative='two-sided')
    return pvalue

def showlossgraph(losses):
    plt.plot(losses, "ro-", label="Train loss")
    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # plt.savefig()
    plt.show()

def savelossgraph(losses,path):
    plt.plot(losses, "ro-", label="Train loss")
    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(path)
    # plt.show()