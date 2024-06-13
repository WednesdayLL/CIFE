import sys

sys.path.append(r"/CIFE")
import argparse
import pickle
from sklearn import preprocessing
from sklearn.metrics import average_precision_score, roc_curve, auc
from random import sample
from codes.buildData import *
from codes.castle_pytorch import *
from codes.utils import *

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.append(r"/Users/wednesday/PycharmProjects/03")

from codes.scarf.scarf.loss import NTXent
from codes.scarf.scarf.model import SCARF

from codes.scarf.example.dataset import ExampleDataset
from codes.scarf.example.utils import dataset_embeddings, train_epoch


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

    model_path = '../model/%s.model' % (cancer)
    scarf_model_path = '../model/PANCAN_scarf.model'


    # 真实数据的处理
    train = pd.read_csv('../dataset/train/%s.csv' % cancer, header=0, index_col=0, sep=',')
    test = pd.read_csv('../dataset/test/%s.csv' % cancer, header=0, index_col=0, sep=',')
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

        for epoch in range(1, 500 + 1):
            epoch_loss = train_epoch(model_scarf, ntxent_loss, train_loader, optimizer, device, epoch)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

        # get embeddings for training and test set
        train_embeddings = dataset_embeddings(model_scarf, train_loader, device)

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
        showlossgraph(losses)
        fp = open(model_path, 'wb')
        # save the trained model
        pickle.dump(classification_model, fp)
        fp.close()
        del classification_model


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

        out_path = '../score/%s_%s.score' % (cancer, emb_dim)
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

        cutoff = 0.57

        gene_sig = df_all.loc[df_all['score'] > cutoff].index.tolist()
        gene_sig = set(gene_sig)
        gene_nsig = df_all.loc[df_all['score'] <= cutoff].index.tolist()
        gene_nsig = set(gene_nsig)

        pd_cgc = pd.read_csv('../input/cgc_somatic.csv', sep=',')
        cgc_key = pd_cgc['Gene'].values.tolist()  # cgc


    elif args.mode == 'embedding_pic':
        # scarf embedding

        model_scarf = pickle.load(open(scarf_model_path, 'rb'))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # get embeddings for training and test set
        train_embeddings = dataset_embeddings(model_scarf, train_loader, device)
        test_embeddings = dataset_embeddings(model_scarf, test_loader, device)

        scatter_train_path = '../picture/embedding_train.png'


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

        out_path = '../pre_score/%s_.score' % (cancer)
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
