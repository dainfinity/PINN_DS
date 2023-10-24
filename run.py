# モデルと設定をimport
import model.PINN as pinn
import config.sol_set as sol_set

# ライブラリをimport
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

print("running...")

# モデルを定義
model = pinn.PINNsModel()

model.to(pinn.device)

params = list(model.parameters())

optimizer = optim.Adam(model.parameters(), lr=0.001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

max_iter = 80000

start_time = time.time()

loss_list = []

N, X0, y0_exa, yprime0_exa, X_min, X_max, f_str, n = sol_set.set()

# 解く
for i in range(max_iter):

    loss = model.loss(N, X0, y0_exa, yprime0_exa, X_min, X_max, f_str, n)
           
    optimizer.zero_grad()     
    
    loss.backward()

    optimizer.step()

    loss_list.append(loss.item())
    
    if i % (max_iter/10) == 0:

        print(loss.item())

elapsed = time.time() - start_time                
print('Training time: %.2f' % (elapsed))

X = torch.tensor(np.linspace(X_min, X_max, N).reshape(N,1))
y_pred = model.predict(X)

X = X.detach().numpy()
y_pred = y_pred.detach().numpy()

plt.plot(X, y_pred)
plt.show()