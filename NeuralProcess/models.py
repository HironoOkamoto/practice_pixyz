from pixyz.distributions import Normal, Deterministic
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# model
class R_encoder(Deterministic):
    def __init__(self, x_dim, y_dim, d_dim, z_dim):
        super(R_encoder, self).__init__(cond_var=["x", "y"], var=["r"])

        self.fc1 = nn.Linear(x_dim+y_dim, d_dim)
        self.fc2 = nn.Linear(d_dim, d_dim)
        self.fc3 = nn.Linear(d_dim, z_dim)

    def forward(self, x, y):
        r = torch.cat([x, y], dim=1)
        r = torch.sigmoid(self.fc1(r))
        r = torch.sigmoid(self.fc2(r))
        r = self.fc3(r)
        return {"r": r}

class S_encoder(Normal):
    def __init__(self, x_dim, y_dim, d_dim, z_dim):
        super(S_encoder, self).__init__(cond_var=["x", "y"], var=["z"])

        self.z_dim = z_dim
        self.fc1 = nn.Linear(x_dim+y_dim, d_dim)
        self.fc2 = nn.Linear(d_dim, d_dim)
        self.fc3 = nn.Linear(d_dim, d_dim)
        self.fc4 = nn.Linear(d_dim, d_dim)
        self.fc5 = nn.Linear(d_dim, z_dim*2)

    def forward(self, x, y):
        z = torch.cat([x, y], dim=1)
        z = torch.sigmoid(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        z = torch.sigmoid(self.fc3(z).mean(0))
        z = torch.sigmoid(self.fc4(z))
        z = self.fc5(z)
        z_mu = z[:self.z_dim]
        z_scale = 0.1 + 0.9*F.softplus(z[self.z_dim:])
        return {"loc": z_mu, "scale": z_scale}

class Decoder(Normal):
    def __init__(self, x_dim, y_dim, d_dim, z_dim, init_func=torch.nn.init.normal_):
        super(Decoder, self).__init__(cond_var=["x_", "r", "z"], var=["y_"])
        self.y_dim = y_dim
        self.fc1 = nn.Linear(x_dim+z_dim*2, d_dim)
        self.fc2 = nn.Linear(d_dim, d_dim)
        self.fc3 = nn.Linear(d_dim, d_dim)
        self.fc4 = nn.Linear(d_dim, y_dim*2)

        if init_func is not None:
            init_func(self.fc1.weight)
            init_func(self.fc2.weight)
            init_func(self.fc3.weight)
            init_func(self.fc4.weight)

    def forward(self, x_, r, z):
        y = torch.cat([x_, r, z], dim=1)
        y = torch.sigmoid(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = torch.sigmoid(self.fc3(y))
        y = self.fc4(y)

        y_mu = y[:, :self.y_dim]
        y_scale = 0.1 + 0.9*F.softplus(y[:, self.y_dim:])
        return {"loc": y_mu, "scale": y_scale}

# single head model
class CrossAttention(Deterministic):
    def __init__(self, x_dim, d_dim, z_dim):
        super(CrossAttention, self).__init__(cond_var=["x_t", "x_c", "r"], var=["r_"])

        self.fc_q = nn.Linear(x_dim, d_dim)
        self.fc_k = nn.Linear(x_dim, d_dim)
        self.fc_v = nn.Linear(z_dim, d_dim)
        self.fc_h = nn.Linear(x_dim, z_dim)

    def forward(self, x_t, x_c, r):
        q = self.fc_q(x_t)
        k = self.fc_k(x_c)
        v = self.fc_v(r)

        sdp = torch.matmul(q, k.t()) / np.sqrt(k.shape[0]) # scaled dot product
        qk = F.softmax(sdp, dim=1)
        head = torch.matmul(qk, v).sum(1).unsqueeze(1)
        return {"r_": self.fc_h(head)}
