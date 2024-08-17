import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# 定义神经网络模型 f(h(t), t) 作为 ODE 函数
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )

    def forward(self, t, h):
        return self.net(h)

# 定义神经ODE模型
class NeuralODE(nn.Module):
    def __init__(self, ode_func):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func

    def forward(self, h0, t):
        return odeint(self.ode_func, h0, t)

# 创建二次函数数据集
t = torch.linspace(0., 2., 100)
true_y = torch.stack([t, t**2], dim=1)

# 初始化神经ODE
ode_func = ODEFunc()
neural_ode = NeuralODE(ode_func)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(neural_ode.parameters(), lr=0.01)

# 训练神经ODE模型
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    pred_y = neural_ode(true_y[0], t)
    loss = criterion(pred_y, true_y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')

# 可视化结果
pred_y = neural_ode(true_y[0], t).detach().numpy()
true_y = true_y.numpy()

plt.plot(true_y[:, 0], true_y[:, 1], 'r', label='True')
plt.plot(pred_y[:, 0], pred_y[:, 1], 'b--', label='Predicted')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()
