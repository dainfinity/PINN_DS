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

def set():
    """
    モデルと設定を定義する関数

    引数:
        なし. \\
        
    戻り値:
        N (int): データ数. \\
        X0 (float): 入力変数の初期値. \\
        y0_exa (float): 初期入力における出力. \\
        yprime0_exa (float): 初期入力における出力の導関数. \\
        X_min (float): 入力の最小値. \\
        X_max (float): 入力の最大値. \\
        f_str (str): 微分方程式. \\
        n (int): 微分方程式の階数. \\
    """
    # データ数
    N = 1000
    # 入力変数の初期値
    X0 = 0
    # 初期入力における出力
    y0_exa = 0
    # 初期入力における出力の導関数
    yprime0_exa = 5.0
    # 入力の最小値
    X_min = 0
    # 出力の最大値
    X_max = 10.0
    # 微分方程式
    f_str = "der[2] + 9.0*der[0]"
    # 微分方程式の階数
    n = 2

    return N, X0, y0_exa, yprime0_exa, X_min, X_max, f_str, n