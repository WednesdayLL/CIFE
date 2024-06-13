import operator
import random
import pandas as pd
import numpy as np

# 造驱动基因和非驱动基因的集子

def feature_drop(x):
    fea = ['gene length','expression_CCLE','replication_time','HiC_compartment','gene_betweeness','gene_degree']
    for i in fea:
        x = x.drop(i,axis=1)
    # print(x.shape)
    return x

def feature_tb(x,fea_nums):
    fea_b = [ "cna_std","5'UTR","expression_CCLE", "lost start and stop","replication_time",]
    fea_t = ['recurrent missense','gene_degree','DEL','SNP','gene_betweeness']
    fea_cna = ['cna_mean','cna_std']
    fea_exp = ['exp_mean','exp_std']
    fea_methy=['methy_mean','methy_std']
    fea_mut = ['silent','nonsense','splice site','missense','recurrent missense','frameshift indel','inframe indel','lost start and stop',
               "3'UTR","5'UTR", 'SNP', 'DEL', 'INS']
    fea_know = ['gene length','expression_CCLE','replication_time','HiC_compartment','gene_betweeness','gene_degree']
    fea_mut_know = fea_mut+fea_know

    df_5 = x
    if fea_nums == 't5':
        df_5 = x[fea_t]
    elif fea_nums == 'b5':
        df_5 = x[fea_b]
    elif fea_nums == 'cna':
        df_5 = x[fea_cna]
    elif fea_nums == 'exp':
        df_5 = x[fea_exp]
    elif fea_nums == 'methy':
        df_5 = x[fea_methy]
    elif fea_nums == 'mut':
        df_5 = x[fea_mut]
    # print(x.shape)
    elif fea_nums == 'know':
        df_5 = x[fea_know]
    elif fea_nums == 'm_k':
        df_5 = x[fea_mut_know]

    return df_5

def build_set(pos_key, neg_key, all_list, nb_imb=10):
    # 驱动基因
    pos_ids = []
    # 非驱动基因
    neg_ids = []
    rand_dis = []
    # print('all_list',all_list)
    for id in all_list:
        gene = id
        if gene in pos_key:
            pos_ids.append(id)
        elif gene in neg_key:
            neg_ids.append(id)
        else:
            rand_dis.append(id)
    rand_dis = random.sample(rand_dis, len(pos_ids) * nb_imb - len(neg_ids))
    # rand_dis = random.sample(rand_dis, len(pos_ids) * nb_imb)
   # 非驱动基因与rand_dis取并集
    neg_ids = list(set(rand_dis) | set(neg_ids))
    # print(neg_ids)
    pos_ids.sort()
    neg_ids.sort()
    # print('驱动基因数量：',len(pos_ids))
    # print('非驱动基因数量：',len(neg_ids))
    return pos_ids, neg_ids

# 训练集数据
def file2data(cancer_type, train_pos, train_neg,mode='normal'):
    # 突变数据
    X_train = []
    X = []

    ## 特征集:
    ## 突变
    if mode == 'normal':
        fea_one = '../feature/%s_train.csv' % (cancer_type)
    else:
        fea_one = '../feature_plus/%s_train.csv' % (cancer_type)
    df_one = pd.read_csv(fea_one, header=0, index_col=0, sep=',')

    feature_name = list(df_one.columns.values)

    # 驱动基因的突变组学数据
    mat_train_pos = df_one.loc[train_pos, ::]
    mat_train_neg = df_one.loc[train_neg, ::]
    gene_name = list(mat_train_pos.index) + list(mat_train_neg.index)
    mat_train_pos = mat_train_pos.values.astype(float)
    mat_train_neg = mat_train_neg.values.astype(float)
    X_train.append(np.concatenate([mat_train_pos, mat_train_neg]))
    X.append(df_one.values.astype(float))
    # label：驱动1 非驱动0
    y_train = np.concatenate([np.ones((len(train_pos))), np.zeros((len(train_neg)))])
    # X_train:list  [array[[...],[...]]] X_train[0] 为np
    # y_train:numpy
    return X_train, y_train, X, feature_name, gene_name

# 测试集数据
def file2test(cancer_type):
    # mode_all = ['mut']
    tumors_set = {'Pancan': 'pancan'}
    X = []
    # 特征集 修改②
    fea_test = '../feature/%s_test.csv' % (cancer_type)
    df_test = pd.read_csv(fea_test, header=0, index_col=0, sep=',')
    feature_name = list(df_test.columns.values)
    gene_name = list(df_test.index)
    X.append(df_test.values.astype(float))
    return X, gene_name,df_test, feature_name

# 输入的特征数据
def feature_input(cancer_type,nb_imb,mode='normal'):
    # 训练集的基因
    if mode=='normal':
        input = '../feature/%s_train.csv' % (cancer_type)
    else:
        input = '../feature_plus/%s_train.csv' % (cancer_type)
    df_fea = pd.read_csv(input, header=0, index_col=0, sep=',')
    train_gene = list(df_fea.index)
    # 突变基因（驱动and非驱动）
    df_gene = pd.read_csv('../input/gene.list', header=0, index_col=0, sep='\t')
    # 基因名字求交集 得到训练集中的突变基因
    # intersection of sets
    all_list = []
    for i in list(df_fea.index):
        if i in list(df_gene.index):
            all_list.append(i)
    # print(len(all_list))
    # all_list = list(set(df_fea.index)&set(df_gene.index))
    # 驱动基因数据集
    key_2020 = '../input/train.txt'
    pd_key = pd.read_csv(key_2020,header=None,sep='\t')
    pd_key.columns = ['gene']
    pd_key = pd_key.drop_duplicates(subset=['gene'],keep='first')
    # 驱动基因
    key_20 = pd_key['gene'].values.tolist()
    # 非驱动基因
    neg_key = ['CACNA1E', 'COL11A1', 'DST', 'TTN']
    # 驱动非驱动的基因名
    key_train_gene = set(key_20)&set(train_gene)
    neg_train_gene = set(neg_key)&set(train_gene)
    # print(neg_train_gene)
    pos, neg = build_set(key_train_gene, neg_train_gene, all_list, nb_imb)
    X_train, y_train, X, feature_name, gene_name = file2data(cancer_type, pos, neg,mode=mode)

    return X_train, y_train, X,  feature_name,gene_name



def evalTest(c, res_file):
    input = "../feature/%s_test.csv" % (c)
    df_tmp = pd.read_csv(input, header=0, index_col=0, sep=',')
    all_list1 = df_tmp.index.tolist()
    pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
    cgc_key = pd_cgc['Gene'].values.tolist()  # cgc
    df_score = pd.read_csv(res_file, index_col=0, header=0, sep=',')
    score_gene = df_score.index.tolist()
    all_list = []
    for i in all_list1:  # test 数据集
        if i in score_gene:  # gene.list
            all_list.append(i)
    pos_cgc, neg_cgc = build_set(cgc_key, [], all_list, nb_imb=10)
    return pos_cgc, neg_cgc




# 输入的特征数据 pancan top5 or bottom5
def feature_input_5(cancer_type,nb_imb=1,fea_nums='t5'):
    input = '../feature/%s_train.csv' % (cancer_type)

    df_fea = pd.read_csv(input, header=0, index_col=0, sep=',')

    train_gene = list(df_fea.index)
    # 突变基因（驱动and非驱动）
    df_gene = pd.read_csv('../input/gene.list', header=0, index_col=0, sep='\t')

    # 基因名字求交集 得到训练集中的突变基因
    # intersection of sets
    all_list = []
    for i in list(df_fea.index):
        if i in list(df_gene.index):
            all_list.append(i)
    # print(len(all_list))
    # all_list = list(set(df_fea.index)&set(df_gene.index))
    # 驱动基因数据集
    key_2020 = '../input/train.txt'
    pd_key = pd.read_csv(key_2020,header=None,sep='\t')
    pd_key.columns = ['gene']
    pd_key = pd_key.drop_duplicates(subset=['gene'],keep='first')
    # 驱动基因
    key_20 = pd_key['gene'].values.tolist()

    # 非驱动基因
    neg_key = ['CACNA1E', 'COL11A1', 'DST', 'TTN']
    # 驱动非驱动的基因名
    key_train_gene = set(key_20)&set(train_gene)
    neg_train_gene = set(neg_key)&set(train_gene)
    # print(neg_train_gene)
    pos, neg = build_set(key_train_gene, neg_train_gene, all_list, nb_imb)
    # 根据基因名得到特征
    X_train, y_train, X, ids, feature_name, gene_name = file2data_5(cancer_type, pos, neg, fea_nums)
    return X_train, y_train, X, ids, feature_name,gene_name

def file2data_5(cancer_type, train_pos, train_neg, fea_nums='t5'):
    # 突变数据
    X_train = []
    X = []
    fea_one = '../feature/%s_train.csv' % (cancer_type)
    df_one = pd.read_csv(fea_one, header=0, index_col=0, sep=',')

    df_one = feature_tb(df_one,fea_nums)
    # df_sim_one = feature_tb(df_sim_one,fea_nums)

    feature_name = list(df_one.columns.values)
    ids = list(df_one.index)
    # ids_sim = list(df_sim_one.index)
    # 驱动基因的突变组学数据
    mat_train_pos = df_one.loc[train_pos, ::]
    mat_train_neg = df_one.loc[train_neg, ::]
    gene_name = list(mat_train_pos.index) + list(mat_train_neg.index)
    mat_train_pos = mat_train_pos.values.astype(float)
    mat_train_neg = mat_train_neg.values.astype(float)

    X_train.append(np.concatenate([mat_train_pos, mat_train_neg]))
    X.append(df_one.values.astype(float))

    y_train = np.concatenate([np.ones((len(train_pos))), np.zeros((len(train_neg)))])
    # X_train:list  [array[[...],[...]]]
    # y_train:numpy
    return X_train, y_train, X, ids,feature_name,gene_name

def file2test_5(cancer_type,fea_nums):
    # mode_all = ['mut']
    tumors_set = {'Pancan': 'pancan'}
    X = []
    # 特征集 修改②
    fea_test = '../feature/%s_test.csv' % (cancer_type)
    df_test = pd.read_csv(fea_test, header=0, index_col=0, sep=',')
    df_test = feature_tb(df_test,fea_nums)
    feature_name = list(df_test.columns.values)
    gene_name = list(df_test.index)
    X.append(df_test.values.astype(float))
    return X, gene_name,df_test, feature_name