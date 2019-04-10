import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pixyz.distributions import Bernoulli, Normal, Deterministic

class MNIST_A_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_A_Classifier, self).__init__()

        self.conv_e = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),    # 64 ⇒ 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),    # 32 ⇒ 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16 ⇒ 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc_e = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1+3+4+10),
        )

    def forward(self, x):
        x = self.conv_e(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc_e(x)
        y1 = torch.sigmoid(x[:, 0])
        y2 = torch.softmax(x[:, 1:4], dim=1)
        y3 = torch.softmax(x[:, 4:8], dim=1)
        y4 = torch.softmax(x[:, 8:], dim=1)
        return y1, y2, y3, y4

class CVAE_Encoder(Normal):
    def __init__(self, z_dim=64, domain_num=1+3+4+10):
        super(CVAE_Encoder, self).__init__(cond_var=["x", "y"], var=["z"])

        self.z_dim = z_dim

        # encode
        self.conv_e = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64 ⇒ 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 ⇒ 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16 ⇒ 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 8 * 8,  40),
        )      
        self.fc2 = nn.Sequential(
            nn.Linear(128 * 8 * 8,  domain_num),
        )        
        
        self.fc = nn.Sequential(
            nn.Linear(40+domain_num, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*self.z_dim),
        )

    def forward(self, x, y):
        x = self.conv_e(x)
        x = x.view(-1, 128 * 8 * 8)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = torch.cat([x1, x2*y], dim=1)
        x = self.fc(x)
        mu = x[:, :self.z_dim]
        scale = F.softplus(x[:, self.z_dim:])
        return {"loc": mu, "scale": scale}


class CVAE_Decoder(Bernoulli):
    def __init__(self, z_dim=64, domain_num=1+3+4+10):
        super(CVAE_Decoder, self).__init__(cond_var=["z", "y"], var=["x"])

        self.z_dim = z_dim

        # decode
        self.fc1 = nn.Sequential(
            nn.Linear(self.z_dim, 40),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.z_dim, domain_num),
        )

        self.fc_d = nn.Sequential(
            nn.Linear(40+domain_num, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128 * 8 * 8),
            nn.LeakyReLU(0.2)
        )
        self.conv_d = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, y):
        z1 = self.fc1(z)
        z2 = self.fc2(z)
        z = torch.cat([z1, z2*y], dim=1)
        h = self.fc_d(z)
        h = h.view(-1, 128, 8, 8)
        return {"probs": self.conv_d(h)}
    
class Encoder_XY(Normal):
    def __init__(self, z_dim=64, y_dim=1+3+4+10):
        super(Encoder_XY, self).__init__(cond_var=["x", "y"], var=["z"])

        self.z_dim = z_dim

        # encode
        self.conv_e = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64 ⇒ 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 ⇒ 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16 ⇒ 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 8 * 8,  40),
        )      
        self.fc2 = nn.Sequential(
            nn.Linear(128 * 8 * 8,  y_dim),
        )        
        
        self.fc = nn.Sequential(
            nn.Linear(40+y_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*self.z_dim),
        )

    def forward(self, x, y):
        x = self.conv_e(x)
        x = x.view(-1, 128 * 8 * 8)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = torch.cat([x1, x2*y], dim=1)
        x = self.fc(x)
        mu = x[:, :self.z_dim]
        scale = F.softplus(x[:, self.z_dim:])
        return {"loc": mu, "scale": scale}
    

class Encoder_X(Normal):
    def __init__(self, z_dim=64):
        super(Encoder_X, self).__init__(cond_var=["x"], var=["z"])

        self.z_dim = z_dim

        # encode
        self.conv_e = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64 ⇒ 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 ⇒ 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16 ⇒ 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*self.z_dim),
        )

    def forward(self, x):
        x = self.conv_e(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc(x)
        mu = x[:, :self.z_dim]
        scale = F.softplus(x[:, self.z_dim:])
        return {"loc": mu, "scale": scale}
    

class Encoder_Y(Normal):
    def __init__(self, z_dim=64, y_dim=1+3+4+10):
        super(Encoder_Y, self).__init__(cond_var=["y"], var=["z"])

        self.z_dim = z_dim

        self.fc = nn.Sequential(
            nn.Linear(y_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*self.z_dim),
        )

    def forward(self, y):
        y = self.fc(y)
        mu = y[:, :self.z_dim]
        scale = F.softplus(y[:, self.z_dim:])
        return {"loc": mu, "scale": scale}
    

class Decoder_X(Bernoulli):
    def __init__(self, z_dim=64):
        super(Decoder_X, self).__init__(cond_var=["z"], var=["x"])

        self.z_dim = z_dim

        # decode
        self.fc_d = nn.Sequential(
            nn.Linear(self.z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128 * 8 * 8),
            nn.LeakyReLU(0.2)
        )
        self.conv_d = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc_d(z)
        h = h.view(-1, 128, 8, 8)
        return {"probs": self.conv_d(h)}

# Deterministicだとloglikelihoodが計算できずに死ぬ？
class Decoder_Y(Bernoulli):
    def __init__(self, z_dim=64, y_dim=18):
        super(Decoder_Y, self).__init__(cond_var=["z"], var=["y"])
        self.y_dim = y_dim
        self.z_dim = z_dim

        # decode
        self.fc_d = nn.Sequential(
            nn.Linear(self.z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, y_dim),
        )

    def forward(self, z):
        y = self.fc_d(z)
        return {"probs":  torch.sigmoid(y)}
