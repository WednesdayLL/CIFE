import sys

sys.path.append(r"/Users/wednesday/PycharmProjects/03")
import argparse
import pickle
from sklearn import preprocessing
from sklearn.metrics import average_precision_score, roc_curve, auc
from random import sample
from codes.buildData import *
from codes.castle_pytorch import *
from codes.utils import *
# from codes.scarf.utils import *
# from codes.scarf.example.dataset import *
from umap import UMAP

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.append(r"/Users/wednesday/PycharmProjects/03")

from codes.scarf.scarf.loss import NTXent
from codes.scarf.scarf.model import SCARF

from codes.scarf.example.dataset import ExampleDataset
from codes.scarf.example.utils import dataset_embeddings, train_epoch

# 扩增模型
from codes.diffusion import *

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 42 28 10 =》0.801 0.392
    parser.add_argument('--reg_lambda', type=float, default=1)
    parser.add_argument('--reg_beta', type=float, default=5)
    parser.add_argument('--mode', type=str, default='embedding_pic')  # train test casual embedding_pic
    parser.add_argument('--type', type=str, default='PANCAN')
    parser.add_argument('--emb_dim', type=int, default=42)  # 42 44
    parser.add_argument('--hz1', type=int, default=28)  # 28 28
    parser.add_argument('--hz2', type=int, default=10)  # 10 10
    parser.add_argument('--hz3', type=int, default=6)
    parser.add_argument('--hz4', type=int, default=6)
    parser.add_argument('--dmodel', type=int, default=32)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dimFF', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=20)
    # parser.add_argument('--lr',type=float,default=0.0014)
    # parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--loss_alpha', type=float, default=0.25)
    parser.add_argument('--loss_gamma', type=int, default=2)
    # scarf
    parser.add_argument('--corruption_rate', type=float, default=0.6)
    parser.add_argument('--encoder_depth', type=int, default=4)
    parser.add_argument('--head_depth', type=int, default=2)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cancer = args.type
    seed = args.seed
    emb_dim = args.emb_dim
    epochs = args.epoch
    hidden_size = [args.hz1, args.hz2]
    # hidden_size = [args.hz1, args.hz2,args.hz3]
    params = {
        'dmodel': args.dmodel,
        'nhead': args.nhead,
        'dimFF': args.dimFF
    }
    params_note = [args.dmodel,
                   args.nhead,
                   args.dimFF]

    batch_size = args.bs
    reg_lambda = args.reg_lambda
    reg_beta = args.reg_beta
    learning_rate = args.lr
    # focal_loss
    loss_alpha = args.loss_alpha
    loss_gamma = args.loss_gamma

    rho_i = 1.0
    alpha_i = 1.0
    w_threshold = 0.3
    share_method = 'stgTrans'

    torch.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # model_path = '../model/%s.model' % (cancer)
    # scarf_model_path = '../model/%s_scarf.model' % (cancer)
    # 调参
    model_path = '../model/%s_%s_%s_%s_%s.model' % (
        cancer, emb_dim, str(hidden_size), learning_rate,
        str([args.corruption_rate, args.encoder_depth, args.head_depth]))
    # scarf_model_path = '../model/%s_%s_%s_scarf.model' % (cancer,emb_dim,str([args.corruption_rate,args.encoder_depth,args.head_depth]))
    scarf_model_path = '../model/PANCAN_scarf.model'
    # scatter_path = '../result/loss/%s_%s.png' % (cancer,emb_dim)

    # df_test = pd.read_csv('../dataset/test/%s.csv'%cancer, header=0, index_col=0, sep=',')
    # # df_test = pd.read_csv('../dataset_02/test/%s.csv' % cancer, header=0, index_col=0, sep=',')
    # # df_test = pd.read_csv('../dataset_plus/test/%s.csv' % cancer, header=0, index_col=0, sep=',')
    # df_test_copy = df_test.copy()

    # print("all the same shape",betas.shape)

    # 演示原始数据分布加噪100步后的结果

    # # 归一化
    # scaler = preprocessing.RobustScaler()
    # df = scaler.fit_transform(df)
    # df_test = scaler.transform(df_test)
    #
    # # X_test = df_test
    # # y_test = df_test[:, 0]
    # X_DAG = df

    # num_nodes = np.shape(X_DAG)[1]
    # print(num_nodes)
    # num_inputs = X_DAG.shape[1]
    # print(num_inputs)

    # X_train = torch.Tensor(X_DAG).float().to(device)
    # y_train = torch.Tensor(X_DAG[:, 0]).float().to(device)
    # X_test = torch.Tensor(X_test).float().to(device)
    # y_test = torch.Tensor(y_test).float().to(device)

    # seed = 1234
    # fix_seed(seed)

    # 真实数据的处理
    # train = pd.read_csv('../dataset/train/%s.csv' % cancer, header=0, index_col=0, sep=',')
    # test = pd.read_csv('../dataset/test/%s.csv' % cancer, header=0, index_col=0, sep=',')
    train = pd.read_csv('../dataset_02/train/%s.csv' % cancer, header=0, index_col=0, sep=',')
    test = pd.read_csv('../dataset_02/test/%s.csv' % cancer, header=0, index_col=0, sep=',')
    feature = train.columns.tolist()

    df_test_copy = test.copy()
    df_train = train.drop('label', axis=1)
    df_test = test.drop('label', axis=1)

    train_target = train['label']
    test_target = test['label']

    scaler = StandardScaler()
    train_data = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns)
    test_data = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

    train_ds = ExampleDataset(
        train_data.to_numpy(),
        train_target.to_numpy(),
        columns=train_data.columns
    )
    test_ds = ExampleDataset(
        test_data.to_numpy(),
        test_target.to_numpy(),
        columns=test_data.columns
    )

    # 在原始数据上训练embedding模型
    if args.mode == 'embedding':
        # to torch dataset

        # print(f"Train set: {train_ds.shape}")
        # print(f"Test set: {test_ds.shape}")
        train_ds.to_dataframe().head()

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model_scarf = SCARF(
            input_dim=train_ds.shape[1],
            emb_dim=emb_dim,
            corruption_rate=args.corruption_rate,
            encoder_depth=args.encoder_depth,
            head_depth=args.head_depth,
        ).to(device)
        optimizer = optim.Adam(model_scarf.parameters(), lr=0.0008)
        ntxent_loss = NTXent()

        # loss_history = []

        for epoch in range(1, 500 + 1):
            epoch_loss = train_epoch(model_scarf, ntxent_loss, train_loader, optimizer, device, epoch)
            # loss_history.append(epoch_loss)

        # showlossgraph(loss_history)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

        # get embeddings for training and test set
        train_embeddings = dataset_embeddings(model_scarf, train_loader, device)

        # tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=15)
        # reduced = tsne.fit_transform(train_embeddings)
        # positive = train_target == 1
        #
        #
        #
        # plt.scatter(reduced[positive, 0], reduced[positive, 1], label="positive")
        # plt.scatter(reduced[~positive, 0], reduced[~positive, 1], label="negative")
        # plt.legend()
        # # plt.show()
        # plt.savefig(scatter_path)

        fp = open(scarf_model_path, 'wb')
        # 存储训练好的模型
        # save the trained model
        pickle.dump(model_scarf, fp)
        fp.close()
        del model_scarf

    elif args.mode == 'train':
        # scarf embedding
        model_scarf = pickle.load(open(scarf_model_path, 'rb'))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # get embeddings for training and test set
        train_embeddings = dataset_embeddings(model_scarf, train_loader, device)
        test_embeddings = dataset_embeddings(model_scarf, test_loader, device)

        # emb_path_train = '../model/%s_train.embedding' % (cancer)
        # emb_path_test = '../model/%s_test.embedding' % (cancer)
        # fp = open(emb_path_train, 'wb')
        # # 存储训练好的模型
        # # save the trained model
        # pickle.dump(model_scarf, fp)
        # fp.close()
        # del model_scarf

        X_train = torch.from_numpy(np.insert(train_embeddings, 0, values=train_target.values.flatten(), axis=1))
        y_train = torch.tensor(np.array(train_target.values.flatten()))

        num_nodes = train_embeddings.shape[1] + 1
        num_inputs = train_embeddings.shape[1] + 1

        print('Training classification model...')
        classification_model = CASTLE(num_inputs=num_inputs, reg_lambda=reg_lambda, reg_beta=reg_beta,
                                      lr=learning_rate, n_hidden=hidden_size, params=params,
                                      batch_size=batch_size, num_outputs=2, DAG_min=0.5, share_method=share_method,
                                      device=device).to(device)
        optimizer = optim.Adam(classification_model.parameters(), lr=learning_rate)
        castleLoss = CastleLoss(num_inputs=num_inputs, num_train=X_train.shape[0],
                                rho=rho_i, alpha=alpha_i, reg_lambda=reg_lambda, reg_beta=reg_beta,
                                loss_alpha=loss_alpha, loss_gamma=loss_gamma, device=device)
        losses = []
        train_loss = []
        for epoch in range(1, epochs):

            # W, Out, pre, weights,_ = model(X_train)
            # supervised_loss, h = castleLoss(y=y_train, pre=pre, X=X_train, W=W, weights=weights, Out=Out)
            # # print("Step " + str(epoch) + ", Loss= " + "{:.4f}".format(supervised_loss), " h_value:", h.detach().cpu().numpy())
            # losses.append(supervised_loss)
            for step in range(1, (X_train.shape[0] // batch_size) + 1):
                idxs = random.sample(range(X_train.shape[0]), batch_size)
                batch_x = X_train[idxs]
                batch_y = torch.Tensor(batch_x[:, 0]).float()
                one_hot_sample = [0] * num_inputs
                subset_ = sample(range(num_inputs), num_nodes)
                for j in subset_:
                    one_hot_sample[j] = 1
                batch_W, batch_Out, batch_pre, batch_weights, _ = classification_model(batch_x)
                supervised_loss, loss_op_dag, h = castleLoss(y=batch_y, pre=batch_pre, X=batch_x, W=batch_W,
                                                             weights=batch_weights, Out=batch_Out,
                                                             sample=one_hot_sample)
                optimizer.zero_grad()
                loss_op_dag.backward(loss_op_dag.clone().detach())
                optimizer.step()
                train_loss.append(supervised_loss.data.numpy())
                # 取一个epoch的loss的平均值，记为这一循环的loss
            losses.append(np.mean(train_loss))
        # showlossgraph(losses)
        fp = open(model_path, 'wb')
        # 存储训练好的模型
        # save the trained model
        pickle.dump(classification_model, fp)
        fp.close()
        del classification_model
        # W_est = W
        # W_est[np.abs(W_est) < w_threshold] = 0
        # showlossgraph(losses)
        # path = '../result/loss/%s_%s_%s_%s_%s.png'%(cancer, str(hidden_size),str(learning_rate),str(epochs),str(batch_size))
        # savelossgraph(losses,path)

    elif args.mode == 'test':
        # scarf embedding
        model_scarf = pickle.load(open(scarf_model_path, 'rb'))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # get embeddings for training and test set
        train_embeddings = dataset_embeddings(model_scarf, train_loader, device)
        test_embeddings = dataset_embeddings(model_scarf, test_loader, device)

        X_train = torch.from_numpy(np.insert(train_embeddings, 0, values=train_target, axis=1))
        y_train = torch.tensor(np.array(train_target))

        print('testing')
        X_test = torch.from_numpy(np.insert(test_embeddings, 0, values=test_target, axis=1))
        y_test = torch.tensor(np.array(test_target))

        model = pickle.load(open(model_path, 'rb'))

        W, _, _, _, out = model(X_train)

        # df_W = pd.DataFrame(W, index=feature, columns=feature)
        # # df_W.to_csv('../result/weight_%d.csv'%int(time.time()))
        # df_W.to_csv('../result/weight.csv')
        # casualPic()
        out = out[:, -1].tolist()
        X_train_casual = df_train.copy()
        X_train_casual.insert(0, 'predict', [1 if val >= 0.5 else 0 for val in out])
        X_train_casual.to_csv('../pre_data/%s_train.csv' % cancer)

        fpr, tpr, _ = roc_curve(y_train, out)
        roc_auc = auc(fpr, tpr)
        print('训练集上测试', roc_auc)

        # 预测
        _, _, _, _, y_test_predict = model(X_test)
        y_test_predict = y_test_predict[:, -1].tolist()
        df_test_copy['predict'] = y_test_predict

        X_test_casual = df_test.copy()
        X_test_casual.insert(0, 'predict', [1 if val >= 0.5 else 0 for val in y_test_predict])
        X_test_casual.to_csv('../pre_data/%s_test.csv' % cancer)

        # score
        genes = set(pd.read_csv('../input/gene.list', header=0, index_col=0, sep='\t').index.tolist())
        df_all = pd.DataFrame(np.nan, index=genes, columns=['score'])
        df_all.loc[df_test_copy.index.tolist(), 'score'] = df_test_copy.loc[df_test_copy.index.tolist(), 'predict']
        df_all['score'].fillna(0, inplace=True)

        # out_path = '../score/%s_%s_%s_%s_%s_%s_%s_%s.score' % (cancer, str(hidden_size),str(learning_rate),str(epochs),str(batch_size),str(seed),str(loss_gamma),str(loss_alpha*1000))
        # out_path = '../score/%s_%s_%s_%s.score' % (cancer, str(hidden_size), str(learning_rate), str(batch_size))
        out_path = '../score/%s_%s_%s_%s.score' % (
            cancer, str(hidden_size), str(learning_rate), emb_dim)
        df_all = df_all.sort_values(by=['score'], ascending=[False])
        df_all.to_csv(out_path, header=True)

        # 预测结果
        pos_cgc, neg_cgc = evalTest(cancer, out_path)
        score_pos_cgc = df_test_copy.loc[pos_cgc, 'predict'].values.tolist()
        score_neg_cgc = df_test_copy.loc[neg_cgc, 'predict'].values.tolist()
        # 真
        y_cgc = np.concatenate([np.ones((len(score_pos_cgc))), np.zeros((len(score_neg_cgc)))])
        # 预测
        y_p_cgc = np.concatenate([score_pos_cgc, score_neg_cgc])
        if cancer == 'PANCAN':
            print('yes')
            gene_cgc = pos_cgc + neg_cgc
            df_PANCAN_gene = pd.DataFrame(gene_cgc, columns=['gene'])
            df_PANCAN_gene.set_index('gene', inplace=True)
            df_PANCAN_gene['score'] = y_p_cgc
            df_PANCAN_gene['class'] = y_cgc
            # use in drawing picture later
            df_PANCAN_gene.to_csv('../PANCAN_test_performance.csv')
        auprc = '%.3f' % (average_precision_score(y_cgc, y_p_cgc))
        fpr, tpr, thresholds = roc_curve(y_cgc, y_p_cgc)
        auroc = '%.3f' % auc(fpr, tpr)
        print("PRAUC =", auprc)
        print("AUROC =", auroc)
        path_cutoff = '../cutoff/cutoff_%s.csv' % (cancer)
        optimal, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
        optimal_th = optimal
        df_cutoff = pd.DataFrame([optimal_th], columns={'cutoff'})
        df_cutoff.to_csv(path_cutoff)

        #### cutoff的富集
        # df = pd.read_csv(path_cutoff, sep=',', index_col=0, header=0)
        # cutoff = df['cutoff'].values.tolist()[0]
        cutoff = 0.57

        gene_sig = df_all.loc[df_all['score'] > cutoff].index.tolist()
        gene_sig = set(gene_sig)
        gene_nsig = df_all.loc[df_all['score'] <= cutoff].index.tolist()
        gene_nsig = set(gene_nsig)

        pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
        cgc_key = pd_cgc['Gene'].values.tolist()  # cgc

        a = len(gene_sig)
        print('卡score基因个数为%d,' % a)

        b = fisher(gene_sig, gene_nsig, cgc_key)
        print('卡score富集：cgc', '%0.1f' % b)

        with open(r'../result/scarf/log1.txt', 'a+') as f:
            f.writelines(
                'learning_rate: ' + str(learning_rate) + '\n' +
                'hidden_size1: ' + str(args.hz1) + '\n' +
                'hidden_size2: ' + str(args.hz2) + '\n' +
                # 'hidden_size3: ' + str(args.hz3) + '\n' +
                # 'hidden_size4: ' + str(args.hz4) + '\n' +
                'params：' + str(params) + '\n' +
                'batch_size: ' + str(batch_size) + '\n' +
                'epoch: ' + str(epochs) + '\n' +
                'seed' + str(seed) + '\n' +
                'embedding' + str(emb_dim) + '\n' +
                'corruption_rate' + str(args.corruption_rate) + '\n' +
                'encoder_depth' + str(args.encoder_depth) + '\n' +
                'head_depth' + str(args.head_depth) + '\n' +
                # 'gamma: ' + str(loss_gamma) + '\n' +
                # 'alpha' + str(loss_alpha) + '\n' +
                '训练集上测试: ' + str(roc_auc) + '\n' +
                'PRAUC = ' + str(auprc) + '\n' +
                'AUROC = ' + str(auroc) + '\n' +
                # '卡score基因个数为: ' + str(a) + '\n' +
                # '卡score富集: ' + str(b) + '\n' +
                '--------------------' + '\n'
            )
            f.close()

    elif args.mode == 'embedding_pic':
        # scarf embedding

        model_scarf = pickle.load(open(scarf_model_path, 'rb'))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # get embeddings for training and test set
        train_embeddings = dataset_embeddings(model_scarf, train_loader, device)
        test_embeddings = dataset_embeddings(model_scarf, test_loader, device)

        scatter_train_path = '../picture/embedding_train.png'

        # TSNE
        # for
        # ,n_iter=2000
        # for m in ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'canberra',
        #                     'braycurtis', 'mahalanobis', 'wminkowski','seuclidean', 'cosine', 'correlation', 'hamming',
        #                     'jaccard', 'dice', 'russellrao', 'kulsinski', 'rogerstanimoto', 'sokalmichener', 'sokalsneath','yule', ]:
        #
        #     perplexity = 20
        #     tsne_train = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=perplexity,n_iter=5000,metric=m)
        #     reduced_train = tsne_train.fit_transform(train_embeddings)
        #
        #
        #
        #     positive_train = train_target == 1
        #     plt.scatter(reduced_train[~positive_train, 0], reduced_train[~positive_train, 1], label="negative")
        #     plt.scatter(reduced_train[positive_train, 0], reduced_train[positive_train, 1], label="positive")
        #     plt.legend()
        #     plt.title('%s_%s'%(perplexity,m))
        #
        #     # plt.savefig(scatter_train_path,dpi=600)
        #     plt.show()

        # umap
        for m in ['euclidean', 'manhattan', 'chebyshev', 'minkowski','canberra',
                  'braycurtis', 'mahalanobis', 'wminkowski','seuclidean', 'cosine', 'correlation', 'hamming',
                  'jaccard', 'dice', 'russellrao', 'kulsinski', 'rogerstanimoto', 'sokalmichener', 'sokalsneath','yule', ]:
            for seed in range(1):
                for neighbor in [50]:
                    # for neighbor in [2, 5, 10, 20, 50, 100, 200]:
                    #     for dist in [0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,]:
                    for dist in [0.4]:
                        reducer = UMAP(n_neighbors=neighbor,
                                       n_components=2,
                                       random_state=seed,
                                       min_dist=dist,
                                       metric=m,
                                       spread=2.0,
                                       init="random",
                                       learning_rate=1,
                                       target_n_neighbors=-1)
                        embedding = reducer.fit_transform(train_embeddings)

                        fig = plt.figure(figsize=(8, 6))
                        # ax = fig.add_subplot(111)
                        # x = embedding[:, 0]
                        # y = embedding[:, 1]
                        plt.title('%s_%s_%s.png' % (seed, neighbor, dist))

                        print(seed)

                        positive_train = train_target == 1

                        plt.scatter(embedding[~positive_train, 0], embedding[~positive_train, 1], label="negative")
                        plt.scatter(embedding[positive_train, 0], embedding[positive_train, 1], label="positive")
                        plt.legend()

                        plt.show()

    elif args.mode == 'casual':
        pre_model_path = '../pre_model/%s.model' % cancer
        train = pd.read_csv('../pre_data/%s_train.csv' % cancer, sep=',', index_col=0, header=0)
        test = pd.read_csv('../pre_data/%s_test.csv' % cancer, sep=',', index_col=0, header=0)

        df_train = train.drop('predict', axis=1)
        df_test = test.drop('predict', axis=1)

        train_target = train['predict']
        test_target = test['predict']

        scaler = preprocessing.StandardScaler()
        df = scaler.fit_transform(df_train)
        df_test = scaler.transform(df_test)

        # X_test = df_test
        # y_test = df_test[:, 0]
        # X_test = torch.Tensor(X_test).float().to(device)
        # y_test = torch.Tensor(y_test).float().to(device)
        #
        # X_DAG = df
        #
        # num_nodes = np.shape(X_DAG)[1]
        # num_inputs = X_DAG.shape[1]
        #
        # X_train = torch.Tensor(X_DAG).float().to(device)
        # y_train = torch.Tensor(X_DAG[:, 0]).float().to(device)
        print(df.shape)
        X_train = torch.from_numpy(np.insert(df, 0, values=train_target.values.flatten(), axis=1)).float()
        y_train = torch.tensor(np.array(train_target.values.flatten()))

        X_test = torch.from_numpy(np.insert(df_test, 0, values=test_target.values.flatten(), axis=1)).float()
        y_test = torch.tensor(np.array(test_target))

        num_nodes = np.shape(X_train)[1]
        num_inputs = X_train.shape[1]

        print('Training casual model...')
        pre_model = CASTLE(num_inputs=num_inputs, reg_lambda=reg_lambda, reg_beta=reg_beta,
                           lr=learning_rate, n_hidden=hidden_size, params=params,
                           batch_size=batch_size, num_outputs=2, DAG_min=0.5, share_method=share_method,
                           device=device).to(device)
        optimizer = optim.Adam(pre_model.parameters(), lr=learning_rate)
        castleLoss = CastleLoss(num_inputs=num_inputs, num_train=X_train.shape[0],
                                rho=rho_i, alpha=alpha_i, reg_lambda=reg_lambda, reg_beta=reg_beta,
                                loss_alpha=loss_alpha, loss_gamma=loss_gamma, device=device)
        losses = []
        train_loss = []
        for epoch in range(1, epochs):
            for step in range(1, (X_train.shape[0] // batch_size) + 1):
                idxs = random.sample(range(X_train.shape[0]), batch_size)
                batch_x = X_train[idxs]
                batch_y = torch.Tensor(batch_x[:, 0]).float()
                one_hot_sample = [0] * num_inputs
                subset_ = sample(range(num_inputs), num_nodes)
                for j in subset_:
                    one_hot_sample[j] = 1
                batch_W, batch_Out, batch_pre, batch_weights, _ = pre_model(batch_x)
                supervised_loss, loss_op_dag, h = castleLoss(y=batch_y, pre=batch_pre, X=batch_x, W=batch_W,
                                                             weights=batch_weights, Out=batch_Out,
                                                             sample=one_hot_sample)
                optimizer.zero_grad()
                loss_op_dag.backward(loss_op_dag.clone().detach())
                optimizer.step()
                train_loss.append(supervised_loss.data.numpy())
                # 取一个epoch的loss的平均值，记为这一循环的loss
            losses.append(np.mean(train_loss))
        # showlossgraph(losses)
        fp = open(pre_model_path, 'wb')
        # 存储训练好的模型
        # save the trained model
        pickle.dump(pre_model, fp)
        fp.close()
        del pre_model

        print('testing')

        model = pickle.load(open(pre_model_path, 'rb'))

        W, _, _, _, out = model(X_train)

        df_W = pd.DataFrame(W.tolist(), index=feature, columns=feature)
        # df_W.to_csv('../result/weight_%d.csv'%int(time.time()))
        df_W.to_csv('../result/weight.csv')
        # casualPic(0)
        out = out[:, -1].tolist()
        fpr, tpr, _ = roc_curve(y_train, out)
        roc_auc = auc(fpr, tpr)
        print('训练集上测试', roc_auc)

        # 预测
        _, _, _, _, y_test_predict = model(X_test)
        y_test_predict = y_test_predict[:, -1].tolist()
        df_test_copy['predict'] = y_test_predict

        # score
        genes = set(pd.read_csv('../input/gene.list', header=0, index_col=0, sep='\t').index.tolist())
        df_all = pd.DataFrame(np.nan, index=genes, columns=['score'])
        df_all.loc[df_test_copy.index.tolist(), 'score'] = df_test_copy.loc[df_test_copy.index.tolist(), 'predict']
        df_all['score'].fillna(0, inplace=True)

        # out_path = '../score/%s_%s_%s_%s_%s_%s_%s_%s.score' % (cancer, str(hidden_size),str(learning_rate),str(epochs),str(batch_size),str(seed),str(loss_gamma),str(loss_alpha*1000))
        # out_path = '../score/%s_%s_%s_%s.score' % (cancer, str(hidden_size), str(learning_rate), str(batch_size))
        out_path = '../pre_score/%s_%s_%s.score' % (
            cancer, str(hidden_size), str(learning_rate))
        df_all = df_all.sort_values(by=['score'], ascending=[False])
        df_all.to_csv(out_path, header=True)

        # 预测结果
        pos_cgc, neg_cgc = evalTest(cancer, out_path)
        score_pos_cgc = df_test_copy.loc[pos_cgc, 'predict'].values.tolist()
        score_neg_cgc = df_test_copy.loc[neg_cgc, 'predict'].values.tolist()
        # 真
        y_cgc = np.concatenate([np.ones((len(score_pos_cgc))), np.zeros((len(score_neg_cgc)))])
        # 预测
        y_p_cgc = np.concatenate([score_pos_cgc, score_neg_cgc])
        auprc = '%.3f' % (average_precision_score(y_cgc, y_p_cgc))
        fpr, tpr, thresholds = roc_curve(y_cgc, y_p_cgc)
        auroc = '%.3f' % auc(fpr, tpr)
        print("PRAUC =", auprc)
        print("AUROC =", auroc)

    # 扩散模型 扩增数据
    # 对真实 未进行归一化的数据进行扩增test_target
    # elif args.mode == 'model':
    #     # 确定超参数的值
    #     dff_batch_size = 16  # 2
    #     dff_num_epoch = 20
    #     dff_num_steps = 98  # 98
    #     # 制定每一步的beta
    #     betas = torch.linspace(-6, 6, dff_num_steps)
    #     betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    #     # 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
    #     alphas = 1 - betas
    #     alphas_prod = torch.cumprod(alphas, 0)
    #     alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    #     alphas_bar_sqrt = torch.sqrt(alphas_prod)
    #     one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    #     one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    #     assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == \
    #            alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape \
    #            == one_minus_alphas_bar_sqrt.shape
    #
    #     # 确定扩散过程任意时刻的采样值 可以基于x[0]得到任意时刻t的x[t]
    #     # def q_x(x_0, t):
    #     #     noise = torch.randn_like(x_0)
    #     #     alphas_t = alphas_bar_sqrt[t]
    #     #     alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    #     #     return (alphas_t * x_0 + alphas_1_m_t * noise)  # 在x[0]的基础上添加噪声
    #
    #     # 开始训练模型，打印loss及中间重构效果
    #     print('Training diffusion model...')
    #     dataset = torch.Tensor(df_train.values).float()
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=dff_batch_size)
    #     # , shuffle=True
    #
    #     dff_model = MLPDiffusion(dff_num_steps, dataset.shape[1])  # 输入是x和step
    #     optimizer = torch.optim.Adam(dff_model.parameters(), lr=0.0000015)  # 0.00005,0.0051
    #     # 初始化
    #     X_train_augmentation = 0
    #     diff_losses = []
    #     for t in range(dff_num_epoch):
    #         diff_loss_epoch = []
    #         i = 0
    #         for idx, batch_x in enumerate(dataloader):
    #
    #             loss = diffusion_loss_fn(dff_model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, dff_num_steps)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(dff_model.parameters(), 1.)
    #             optimizer.step()
    #             diff_loss_epoch.append(loss)
    #             i = idx
    #         # if(t%2==0):
    #         diff_losses.append(sum(diff_loss_epoch)/(i+1))
    #
    #         if (t == dff_num_epoch - 1):
    #             x_seq = p_sample_loop(dff_model, dataset.shape, dff_num_steps, betas, one_minus_alphas_bar_sqrt)
    #             length = len(x_seq)
    #             X_train_augmentation = x_seq[length - 1]
    #             print(X_train_augmentation.shape, type(X_train_augmentation))
    #     showlossgraph(diff_losses)
    #     X_train_augmentation = X_train_augmentation.detach().numpy()
    #     y_train_augmentation = train_target
    #
    #
    #     # 扩增数据scaler
    #
    #     X_train_augmentation = scaler.fit_transform(X_train_augmentation)
    #     # 合并真实数据与扩增数据
    #     # print(df_train,type(df_train),X_train_augmentation,type(X_train_augmentation))
    #     dff_train_x = np.concatenate([df_train, X_train_augmentation], axis=0)
    #     dff_train_y = np.concatenate([train_target, y_train_augmentation], axis=0)
    #
    #     dff_train_x = pd.DataFrame(dff_train_x)
    #     dff_train_y = pd.DataFrame(dff_train_y)
    #
    #
    #     # to torch dff dataset
    #     dff_train_ds = ExampleDataset(
    #         dff_train_x.to_numpy(),
    #         dff_train_y.to_numpy(),
    #         # columns=train_data.columns
    #     )
    #     test_ds = ExampleDataset(
    #         test_data.to_numpy(),
    #         test_target.to_numpy(),
    #         columns=test_data.columns
    #     )
    #     # scarf embedding
    #     model_scarf = pickle.load(open(scarf_model_path, 'rb'))
    #     dff_train_loader = DataLoader(dff_train_ds, batch_size=batch_size, shuffle=False)
    #     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    #
    #     # get embeddings for training and test set
    #     train_embeddings = dataset_embeddings(model_scarf, dff_train_loader, device)
    #     test_embeddings = dataset_embeddings(model_scarf, test_loader, device)
    #
    #     # emb_path_train = '../model/%s_train.embedding' % (cancer)
    #     # emb_path_test = '../model/%s_test.embedding' % (cancer)
    #     # fp = open(emb_path_train, 'wb')
    #     # # 存储训练好的模型
    #     # # save the trained model
    #     # pickle.dump(model_scarf, fp)
    #     # fp.close()
    #     # del model_scarf
    #
    #
    #
    #     X_train = torch.from_numpy(np.insert(train_embeddings, 0, values=dff_train_y.values.flatten(), axis=1))
    #     y_train = torch.tensor(np.array(dff_train_y.values.flatten()))
    #
    #
    #     num_nodes = train_embeddings.shape[1]+1
    #     num_inputs = train_embeddings.shape[1]+1
    #
    #     print('Training classification model...')
    #     classification_model = CASTLE(num_inputs=num_inputs, reg_lambda=reg_lambda,reg_beta=reg_beta,
    #                     lr=learning_rate, n_hidden=hidden_size,params=params,
    #                     batch_size=batch_size, num_outputs=2, DAG_min=0.5, share_method=share_method,device=device).to(device)
    #     optimizer = optim.Adam(classification_model.parameters(), lr=learning_rate)
    #     castleLoss = CastleLoss(num_inputs=num_inputs,  num_train=X_train.shape[0],
    #                             rho=rho_i, alpha=alpha_i, reg_lambda=reg_lambda, reg_beta=reg_beta,loss_alpha=loss_alpha,loss_gamma=loss_gamma,device=device)
    #     losses = []
    #     train_loss=[]
    #     for epoch in range(1, epochs):
    #
    #         # W, Out, pre, weights,_ = model(X_train)
    #         # supervised_loss, h = castleLoss(y=y_train, pre=pre, X=X_train, W=W, weights=weights, Out=Out)
    #         # # print("Step " + str(epoch) + ", Loss= " + "{:.4f}".format(supervised_loss), " h_value:", h.detach().cpu().numpy())
    #         # losses.append(supervised_loss)
    #         for step in range(1, (X_train.shape[0] // batch_size) + 1):
    #             idxs = random.sample(range(X_train.shape[0]), batch_size)
    #             batch_x = X_train[idxs]
    #             batch_y = torch.Tensor(batch_x[:, 0]).float()
    #             one_hot_sample = [0] * num_inputs
    #             subset_ = sample(range(num_inputs), num_nodes)
    #             for j in subset_:
    #                 one_hot_sample[j] = 1
    #             batch_W, batch_Out, batch_pre, batch_weights, _ = classification_model(batch_x)
    #             supervised_loss,loss_op_dag,h = castleLoss(y=batch_y, pre=batch_pre, X=batch_x, W=batch_W, weights=batch_weights, Out=batch_Out, sample=one_hot_sample)
    #             optimizer.zero_grad()
    #             loss_op_dag.backward(loss_op_dag.clone().detach())
    #             optimizer.step()
    #             train_loss.append(supervised_loss.data.numpy())
    #             # 取一个epoch的loss的平均值，记为这一循环的loss
    #         losses.append(np.mean(train_loss))
    #     showlossgraph(losses)
    #     fp = open(model_path, 'wb')
    #     # 存储训练好的模型
    #     # save the trained model
    #     pickle.dump(classification_model, fp)
    #     fp.close()
    #     del classification_model
    #     # W_est = W
    #     # W_est[np.abs(W_est) < w_threshold] = 0
    #     # showlossgraph(losses)
    #     # path = '../result/loss/%s_%s_%s_%s_%s.png'%(cancer, str(hidden_size),str(learning_rate),str(epochs),str(batch_size))
    #     # savelossgraph(losses,path)
    #
    #
    # # elif args.mode == 'test':
    #
    #     # X_train = torch.from_numpy(np.insert(train_embeddings, 0, values=train_target, axis=1))
    #     # y_train = torch.tensor(np.array(train_target))
    #     print('testing')
    #     X_test = torch.from_numpy(np.insert(test_embeddings, 0, values=test_target, axis=1))
    #     y_test = torch.tensor(np.array(test_target))
    #
    #     model = pickle.load(open(model_path, 'rb'))
    #
    #     W,_,_,_,out = model(X_train)
    #     # df_W = pd.DataFrame(W, index=feature, columns=feature)
    #     # # df_W.to_csv('../result/weight_%d.csv'%int(time.time()))
    #     # df_W.to_csv('../result/weight.csv')
    #     out = out[:, -1].tolist()
    #     fpr, tpr, _ = roc_curve(y_train, out)
    #     roc_auc = auc(fpr, tpr)
    #     print('训练集上测试',roc_auc)
    #
    #     # 预测
    #     _,_,_,_,y_test_predict = model(X_test)
    #     y_test_predict =y_test_predict[:, -1].tolist()
    #     df_test_copy['predict'] = y_test_predict
    #
    #
    #     # score
    #     genes = set(pd.read_csv('../input/gene.list', header=0, index_col=0, sep='\t').index.tolist())
    #     df_all = pd.DataFrame(np.nan, index=genes, columns=['score'])
    #     df_all.loc[df_test_copy.index.tolist(), 'score'] = df_test_copy.loc[df_test_copy.index.tolist(), 'predict']
    #     df_all['score'].fillna(0, inplace=True)
    #
    #     # out_path = '../score/%s_%s_%s_%s_%s_%s_%s_%s.score' % (cancer, str(hidden_size),str(learning_rate),str(epochs),str(batch_size),str(seed),str(loss_gamma),str(loss_alpha*1000))
    #     # out_path = '../score/%s_%s_%s_%s.score' % (cancer, str(hidden_size), str(learning_rate), str(batch_size))
    #     out_path = '../score/%s_%s_%s_%s.score' % (
    #     cancer, str(hidden_size), str(learning_rate), emb_dim)
    #     df_all = df_all.sort_values(by=['score'], ascending=[False])
    #     df_all.to_csv(out_path, header=True)
    #
    #     # 预测结果
    #     pos_cgc, neg_cgc = evalTest(cancer,out_path)
    #     score_pos_cgc = df_test_copy.loc[pos_cgc, 'predict'].values.tolist()
    #     score_neg_cgc = df_test_copy.loc[neg_cgc, 'predict'].values.tolist()
    #     # 真
    #     y_cgc = np.concatenate([np.ones((len(score_pos_cgc))), np.zeros((len(score_neg_cgc)))])
    #     # 预测
    #     y_p_cgc = np.concatenate([score_pos_cgc, score_neg_cgc])
    #     auprc = '%.3f' % (average_precision_score(y_cgc, y_p_cgc))
    #     fpr, tpr, thresholds = roc_curve(y_cgc, y_p_cgc)
    #     auroc = '%.3f' % auc(fpr, tpr)
    #     print("PRAUC =", auprc)
    #     print("AUROC =", auroc)
    #     # path_cutoff = '../cutoff/cutoff_%s.csv' % (cancer)
    #     # optimal, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    #     # optimal_th = optimal
    #     # df_cutoff = pd.DataFrame([optimal_th], columns={'cutoff'})
    #     # df_cutoff.to_csv(path_cutoff)
    #
    #
    #     #
    #     # #### cutoff的富集
    #     # df = pd.read_csv(path_cutoff, sep=',', index_col=0, header=0)
    #     # cutoff = df['cutoff'].values.tolist()[0]
    #     #
    #     # gene_sig = df_all.loc[df_all['score'] > cutoff].index.tolist()
    #     # gene_sig = set(gene_sig)
    #     # gene_nsig = df_all.loc[df_all['score'] <= cutoff].index.tolist()
    #     # gene_nsig = set(gene_nsig)
    #     #
    #     # pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
    #     # cgc_key = pd_cgc['Gene'].values.tolist()  # cgc
    #     #
    #     # a = len(gene_sig)
    #     # print('卡score基因个数为%d,' % a)
    #     #
    #     # b = fisher(gene_sig, gene_nsig, cgc_key)
    #     # print('卡score富集：cgc', '%0.1f' % b)
    #
    #     with open(r'../result/scarf/log.txt', 'a+') as f:
    #         f.writelines(
    #             'learning_rate: ' + str(learning_rate) + '\n' +
    #             'hidden_size1: ' + str(args.hz1) + '\n' +
    #             'hidden_size2: ' + str(args.hz2) + '\n' +
    #             # 'hidden_size3: ' + str(args.hz3) + '\n' +
    #             # 'hidden_size4: ' + str(args.hz4) + '\n' +
    #             'params：' + str(params) + '\n' +
    #             'batch_size: ' + str(batch_size) + '\n' +
    #             'epoch: ' + str(epochs) + '\n' +
    #             'seed'+str(seed)+'\n'+
    #             'embedding'+str(emb_dim)+'\n'+
    #             # 'gamma: ' + str(loss_gamma) + '\n' +
    #             # 'alpha' + str(loss_alpha) + '\n' +
    #             '训练集上测试: ' + str(roc_auc) + '\n' +
    #             'PRAUC = ' + str(auprc) + '\n' +
    #             'AUROC = ' + str(auroc) + '\n' +
    #             # '卡score基因个数为: ' + str(a) + '\n' +
    #             # '卡score富集: ' + str(b) + '\n' +
    #             '--------------------' + '\n'
    #         )
    #         f.close()
