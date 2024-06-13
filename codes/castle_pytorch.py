import sys

sys.path.append(r"/Users/wednesday/PycharmProjects/03")

import numpy as np

np.set_printoptions(suppress=True)
import torch
from torch import nn
from typing import Optional
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from .utils import showlossgraph
from .focalLoss import focal_loss
from .transformer import Transformer
from .ft_transformer import FTTransformer
from .demo_tab import TabTransformer
# from stg import STG
# from .stg.models import STGClassificationModel
# from .DeepGBM.train_models import GBDT2NN
from .scarf.scarf.model import SCARF
from  .scarf.scarf.loss import NTXent
from .stgTrans import FeatureSelector, STGTrans
from .dual_focal_loss import DualFocalLoss


class CASTLE(nn.Module):
    def __init__(self, n_hidden, num_outputs, num_inputs, lr, batch_size,params,device,
                  reg_lambda=1., reg_beta=1, DAG_min=0.5, share_method='Transformer',):
        super().__init__()
        # self.w_threshold = w_threshold
        self.DAG_min = DAG_min
        self.learning_rate = lr
        # l2正则化
        # self.reg_lambda = reg_lambda
        # self.reg_beta = reg_beta
        self.hidden_layers = len(n_hidden)
        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.metric = roc_auc_score
        self.share_method = share_method
        # self.weights bias{} 字典 其中对于每个特征有自己的第一个weights,和最后一个用来分类的权重和bias
        self.weights = {}
        # self.biases = nn.ParameterDict()
        self.hidden_mask = nn.ModuleDict()
        self.willPredict = nn.ModuleDict()
        for i in range(self.num_inputs):
            # ① 通过将权重相应行置0，而后得到被mask掉的特征 的一层
            self.hidden_mask['w_h0_'+str(i)] = MaskUnit(n_hidden[0], self.num_inputs,device)
            # ③ 预测层
            self.willPredict['out_'+str(i)] = OutUnit(n_hidden[-1], self.num_outputs,device)

        # ② 共享层，可在这层进行创新
        # transformer
        if self.share_method == 'Transformer':
            self.share = Transformer(n_hidden)
        #     加入STG的特征选择和正则
        elif self.share_method == 'stgTrans':
            self.share = STGTrans(n_hidden,0.1,device)
        elif self.share_method == 'tabTransformer':
            self.share = TabTransformer(input_size=n_hidden[0],output_size=n_hidden[1],d_model=params['dmodel'],nhead=params['nhead'],dim_feedforward=params['dimFF'],dropout=0.)
        elif self.share_method == 'ftTransformer':
            self.share = FTTransformer(
                categories = (),      # tuple containing the number of unique values within each category
                num_continuous = n_hidden[0],                # number of continuous values
                dim = 32,                           # dimension, paper set at 32
                dim_out = n_hidden[0],                        # binary prediction, but could be anything
                depth = 6,                          # depth, paper recommended 6
                heads = 8,                          # heads, paper recommends 8
                attn_dropout = 0.1,                 # post-attention dropout
                ff_dropout = 0.1                    # feed forward dropout
            )

        else:
            self.share = nn.Linear(n_hidden[0],n_hidden[1])


        self.hidden_h0 = {}
        self.out_layer = {}
        self.out = {}
        # 中间层激活函数
        # self.activation = nn.ReLU()
        self.activation = nn.GELU()
        # 预测层激活函数
        self.finalCatActivate = torch.softmax

    def forward(self, x):
        # Out_0 = nn.ParameterList()
        Out_0 = []
        # for i in range(1):
        for i in range(self.num_inputs):
            # print(i)
            # h0:输入特征*mask的权重+偏置  第一个
            h0,self.weights['w_h0_'+str(i)] = self.hidden_mask['w_h0_'+str(i)](x,index=i)
            # print('w',self.weights['w_h0_'+str(i)])
            # 每遮盖掉一个特征，剩下的25维特征都有一个自己的第一层隐含层
            self.hidden_h0['nn_' + str(i)] = self.activation(h0)
            # print('w', self.weights['w_h0_' + str(i)])
            # print(self.hidden_h0['nn_' + str(i)].shape)

            # if(self.share_method == 'MLP-Mixer'):


            x_i = self.share(self.hidden_h0['nn_' + str(i)])

            self.out_layer['out_' + str(i)] = self.willPredict['out_' + str(i)](x_i)
            # self.out_layer['out_' + str(i)] = self.willPredict['out_' + str(i)](self.hidden_h0['nn_' + str(i)])

            self.out['out_' + str(i)] = self.finalCatActivate(self.out_layer['out_' + str(i)], dim=1)
            Out_0.append(self.out['out_' + str(i)][:, -1].reshape(-1, 1))
        Out = torch.cat(Out_0, dim=1)
        self.regularization_loss = 0
        # 得到了邻接矩阵 W:(d+1)*(d+1) 对角是0
        self.W_0 = []
        for i in range(self.num_inputs):
            self.W_0.append(torch.sqrt(torch.sum(torch.square(self.weights['w_h0_' + str(i)]), dim=1, keepdim=True)))
        self.W = torch.cat(self.W_0, dim=1)
        # print('邻接矩阵：',self.W)
        # 邻接矩阵W，收集了所有经过激活函数的输出，要进行交叉熵的未经过激活函数的没包含label特征的输出
        return self.W,Out,self.out_layer['out_0'],self.weights,self.out['out_0']

class MaskUnit(nn.Module):
    def __init__(self, n_hidden, num_inputs,device):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_inputs = num_inputs
        # self.indices = [1] * n_hidden
        self.indices = [1e-12] * n_hidden
        self.device=device
        # weight矩阵形状 num_inputs * n_hidden / n_hidden * num_inputs
        # tf.random_normal 此处默认值mean0 标准差std1 标准正态分布
        self.weights = nn.Parameter(torch.tensor(np.random.normal(size=[num_inputs, n_hidden]) * 0.01, requires_grad=True,dtype=torch.float32).to(device),requires_grad=True)
        self.biases = nn.Parameter(torch.tensor(np.random.normal(size=[n_hidden]) * 0.01, requires_grad=True,dtype=torch.float32).to(device),requires_grad=True)

    def forward(self, X, index):
        mask = torch.tensor(np.insert(np.ones([self.num_inputs - 1, self.n_hidden]), index, self.indices, axis=0),dtype=torch.float32).to(self.device)
        weights = torch.mul(self.weights, mask)
        hidden_h0 = torch.add(torch.matmul(X, weights), self.biases)
        return hidden_h0,weights

class OutUnit(nn.Module):
    def __init__(self,n_hidden, num_outputs,device):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(np.random.normal(size=[n_hidden, num_outputs]),dtype=torch.float32, requires_grad=True).to(device),requires_grad=True)
        self.biases = nn.Parameter(torch.tensor(np.random.normal(size=[num_outputs]),dtype=torch.float32, requires_grad=True).to(device))

    def forward(self,final_hidden_output):
        out_layer = torch.matmul(final_hidden_output,self.weights) + self.biases
        return out_layer

class CastleLoss(nn.Module):
    def __init__(self, num_inputs, num_train, rho, alpha, reg_lambda, reg_beta,device,loss_alpha=None,loss_gamma=None,stgLam=0.1,stgSigma=1.0,):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_train = num_train
        self.rho = rho
        self.alpha = alpha
        self.Lambda = reg_lambda
        self.reg_beta = reg_beta
        self.device= device

        # TODO:增加stg loss正则部分
        self.FeatureSelector = FeatureSelector(num_inputs, stgSigma,device)
        self.reg = self.FeatureSelector.regularizer
        self.lam = stgLam
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma


        # 0️⃣ 预测函数，内部自动加softmax
        # 交叉熵函数
        # TODO:修改了损失函数
        # self.loss_pre = nn.CrossEntropyLoss()

        # dual-focalloss函数
        self.loss_pre = DualFocalLoss()


        # focalloss函数
        # self.loss_pre = focal_loss(loss_alpha,loss_gamma)


    def forward(self, y, pre, X, W, weights, Out, sample=None):
        # print('Www',W.grad)
        # supervised_loss = self.loss_pre(pre,y.long())
        # TODO:修改loss，增加正则项
        classification_loss = self.loss_pre(pre,y.long())
        reg = torch.mean(self.reg((self.mu + 0.5) / self.sigma))
        supervised_loss = classification_loss + self.lam * reg
        # supervised_loss = classification_loss

        '''
        ① 求 Rw = (Tr(e^(WoW))-d-1)^2 部分（没平方）
        # truncated power series
        # 特征+1 相当于论文中的d+1
        '''
        d = torch.from_numpy(np.array(X.shape[1])).float().to(self.device)
        coff = 1.0
        Z = torch.multiply(W, W)  # 相同位置的元素相乘 哈达玛积
        # d和 dag_l 相等
        dag_l = torch.from_numpy(np.array(X.shape[1])).float().to(self.device)
        #  Tr(e^(WoW)
        Z_in = torch.eye(X.shape[1],dtype=torch.float32).to(self.device)  # 对角线值为1，其余为0
        # print(Z.dtype)
        for i in range(1, 10):
            Z_in = torch.matmul(Z_in, Z)
            dag_l += 1. / coff * torch.trace(Z_in).to(self.device)
            coff = coff * (i + 1)
        # dag_l = torch.trace(torch.exp(Z))
        # 是指  Rw = (Tr(e^(WoW))-d-1)^2   DAG loss中的R 无平方（辅助loss）  所以后续用了self.h*self.h
        h = dag_l - d

        if(sample==None):
            return supervised_loss,h
        else:
            '''② group lasso DAG loss中的V（l1 loss）'''
            L1_loss = 0.0
            for i in range(self.num_inputs):
                w_1 = weights['w_h0_' + str(i)][0:i,:-1]
                w_2 = weights['w_h0_' + str(i)][i+1:,:-1]
                L1_loss += torch.sum(w_1.norm(p=2, dim=1)) + torch.sum(w_2.norm(p=2, dim=1))
            '''
            ③ 重构损失 maybe
            (   L=1/N|| X-f(X) ||^2   )   DAG loss中的L（重构loss）
            '''
            R = X - Out
            # Average reconstruction loss 没用到
            self.average_loss = 0.5 / self.num_train * torch.sum(torch.square(R))
            # group lasso DAG loss中的V（l1 loss）
            # Divide the residual into untrain and train subset
            sample = torch.tensor(sample,dtype=torch.float32).to(device=self.device)
            _, subset_R = dynamic_partition(R.transpose(1,0), partitions=sample, num_partitions=2)
            subset_R = subset_R.transpose(1,0)

            # Combine all the loss
            mse_loss_subset = torch.Tensor(self.num_inputs).to(device=self.device) / torch.sum(sample) * torch.sum(torch.square(subset_R))
            regularization_loss_subset = mse_loss_subset + self.reg_beta * L1_loss + 0.5 * self.rho * h * h + self.alpha * h
            # Add in supervised loss
            regularization_loss_subset += self.Lambda * self.rho * supervised_loss
            # self.loss_op_dag = self.optimizer_subset.minimize(self.regularization_loss_subset)
            # self.loss_op_supervised = self.optimizer_subset.minimize(self.supervised_loss + self.regularization_loss)
            loss_op_dag = regularization_loss_subset
            regularization_loss = 0
            loss_op_supervised = supervised_loss + regularization_loss
            return supervised_loss,regularization_loss_subset,h

# pytorch实现
def dynamic_partition(data, partitions, num_partitions):
    # print(type(data),type(partitions),num_partitions)
    # Create a list of indices for each partition
    indices = [torch.nonzero(partitions == i)[:, 0] for i in range(num_partitions)]
    # Split the data tensor into a list of tensors based on the indices
    partitions = [torch.index_select(data, dim=0, index=index) for index in indices]
    return partitions

