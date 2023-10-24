# ライブラリをインポート
import torch
import torch.autograd as autograd         
from torch import Tensor                  
import torch.nn as nn                    
import torch.optim as optim

import numpy as np
import time
import scipy.io

import matplotlib.pyplot as plt
import japanize_matplotlib

# デフォルトのデータ型をfloatに設定
torch.set_default_dtype(torch.float)

# 乱数を固定値で初期化
torch.manual_seed(1234)
np.random.seed(1234)

# デバイスを設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PINNsModel(nn.Module):

    def __init__(self):
        super().__init__()

        # 層を定義
        self.fc1_20 = nn.Linear(1,20)
        self.fc20_20 = nn.Linear(20,20)
        self.fc20_1 = nn.Linear(20,1)
        self.tanh = nn.Tanh()

        # 損失関数を定義
        self.loss_function = nn.MSELoss(reduction ='mean')

        # 重みを初期化
        nn.init.xavier_normal_(self.fc1_20.weight.data, gain=1.0)
        nn.init.xavier_normal_(self.fc20_20.weight.data, gain=1.0)
        nn.init.xavier_normal_(self.fc20_1.weight.data, gain=1.0)

        # バイアスを初期化
        nn.init.zeros_(self.fc1_20.bias.data)
        nn.init.zeros_(self.fc20_20.bias.data)
        nn.init.zeros_(self.fc20_1.bias.data)

    def forward(self,X:torch.Tensor, x_min: float, x_max: float):
        """
        順伝播を計算するメソッド

        引数:
            X (torch.Tensor): 入力変数. \\
            x_min (float): 入力変数の最小値. \\
            x_max (float): 入力変数の最大値. \\
        
        戻り値:
            out (torch.Tensor): 出力変数. \\
        """
        if torch.is_tensor(X) != True:         
            X = torch.from_numpy(X).float().to(device).clone()
        
        
        x_min = x_min
        x_max = x_max

        X_min_ts = x_min + torch.zeros(X.shape).to(device)
        X_max_ts = x_max + torch.zeros(X.shape).to(device)

        #入力の前処理
        X_s = (X - X_min_ts)/(X_max_ts - X_min_ts)  #正規化

        #float型に変換
        X_s = X_s.float()

        # 順伝播
        out = self.fc1_20(X_s)

        for i in range(7):
            out = self.tanh(out)
            out = self.fc20_20(out)
            
        out = self.fc20_1(out) 

        return out


    def loss(self, N: int, X0: float, y0_exa: float, yprime0_exa: float, X_min: float, X_max: float, f_str: str, n: int):
        """
        モデルの損失関数を計算するメソッド

        引数:
            N (int): データ数. \\
            X0 (float): 入力変数の初期値. \\
            y0_exa (float): 出力の初期値(教師データ). \\
            yprime0_exa (float): 出力の1階微分の初期値(教師データ). \\
            X_min (float): 入力変数の最小値. \\
            X_max (float): 入力変数の最大値. \\
            f_str (str): 解きたい微分方程式を、全て左辺に持ってきたもの(f=0を解くことになる). \\
            n (int): 微分方程式の階数. \\

        戻り値:
            loss(torch.Tensor): モデルの損失関数の合計値
        """

        self.X_min = X_min
        self.X_max = X_max
        
        # 初期値をtensor型に変換
        X0 = X0 + torch.zeros(N,1).to(device)
        y0_exa = y0_exa + torch.zeros(X0.shape).to(device)
        yprime0_exa = yprime0_exa + torch.zeros(X0.shape).to(device)

        # X_minからX_maxの間でランダムにXを生成
        X = X_min + (X_max - X_min) * torch.rand(X0.shape).to(device)

        # t=0での微分を計算
        h = X0.clone()

        h.requires_grad = True

        y0 = self.forward(h, X_min, X_max)

        y0_x = autograd.grad(y0,h,torch.ones(X0.shape).to(device), retain_graph=True, create_graph=True)[0]

        # t=0以外での微分を計算
        g = X.clone()
                            
        g.requires_grad = True
        
        y = self.forward(g, X_min, X_max)

        y_x = autograd.grad(y,g,torch.ones(X.shape).to(device), retain_graph=True, create_graph=True)[0]

        y_xx = autograd.grad(y_x,g,torch.ones(X.shape).to(device), create_graph=True)[0]

        # 高階微分を計算
        der = self.higherOrderDerivative(y,g,n)
    
        # 微分方程式(der[i]はi階微分)
        f = eval(f_str) # 単振動の微分方程式

        # 初速度についてのloss
        loss_v0 = self.loss_function(y0_x, yprime0_exa)
        # 初期位置に対するloss
        loss_x0 = self.loss_function(y0, y0_exa)  
        # 微分方程式についてのloss
        f0 = torch.zeros(f.shape).to(device)

        loss_f = self.loss_function(f,f0)
        # lossの合計
        loss = loss_v0 + loss_x0 + loss_f

        return loss
    
    def predict(self, X: torch.Tensor):
        """
        予測を行うメソッド

        引数:
            X (torch.Tensor): 任意の入力変数. \\
        
        戻り値:
            y_pred (torch.Tensor): 予測値. \\
        """

        y_pred = self.forward(X, self.X_min, self.X_max)

        return y_pred

    def higherOrderDerivative(self, y: torch.Tensor, x:torch.Tensor, n):
        """
        高階微分を計算するメソッド

        引数:
            y (torch.Tensor): 入力変数. \\
            x (torch.Tensor): 任意の入力変数. \\
            n (int): 高階微分の階数. \\

        戻り値:
            der_list (list): 高階微分のリスト(i番目にi階微分が格納されている). \\
        """

        der_list = [y]
        for i in range(n+1):
            y = autograd.grad(y,x,torch.ones([x.shape[0], 1]).to(device), create_graph=True)[0]
            der_list.append(y)
        return der_list
    