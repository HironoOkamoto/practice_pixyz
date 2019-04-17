import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pixyz.distributions import Normal, Bernoulli, Categorical
    
    
#### JMVAEモデル

class Encoder_XY(Normal):
    def __init__(self, z_dim=2, y_dim=2):
        super(Encoder_XY, self).__init__(cond_var=["x", "y1", "y2"], var=["z"])

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
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8 ⇒ 4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),            
        )
        self.fc_y = nn.Sequential(
            nn.Linear(y_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),                   
        )      
       
        self.fc = nn.Sequential(
            nn.Linear(1024+256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, z_dim*2),
        )

    def forward(self, x, y1, y2):
        x = self.conv_e(x)
        x = x.view(-1, 64 * 4 *4)
        y = self.fc_y(torch.cat([y1, y2], dim=1))
        x = torch.cat([x, y], dim=1)
        x = self.fc(x)
        mu = x[:, :self.z_dim]
        scale = F.softplus(x[:, self.z_dim:])
        return {"loc": mu, "scale": scale}
    

class Encoder_X(Normal):
    def __init__(self, z_dim=2):
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
    

class Encoder_Y1(Normal):
    def __init__(self, z_dim=2, y_dim=1):
        super(Encoder_Y1, self).__init__(cond_var=["y1"], var=["z"])

        self.z_dim = z_dim

        self.fc = nn.Sequential(
            nn.Linear(y_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 2*self.z_dim),
        )

    def forward(self, y1):
        y = self.fc(y1)
        mu = y[:, :self.z_dim]
        scale = F.softplus(y[:, self.z_dim:])
        return {"loc": mu, "scale": scale}
    
class Encoder_Y2(Normal):
    def __init__(self, z_dim=2, y_dim=1):
        super(Encoder_Y2, self).__init__(cond_var=["y2"], var=["z"])

        self.z_dim = z_dim

        self.fc = nn.Sequential(
            nn.Linear(y_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 2*self.z_dim),
        )

    def forward(self, y2):
        y = self.fc(y2)
        mu = y[:, :self.z_dim]
        scale = F.softplus(y[:, self.z_dim:])
        return {"loc": mu, "scale": scale}


class Decoder_X(Bernoulli):
    def __init__(self, z_dim=2):
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


class Decoder_Y1(Bernoulli):
    def __init__(self, z_dim=2, y_dim=1):
        super(Decoder_Y1, self).__init__(cond_var=["z"], var=["y1"])
        self.y_dim = y_dim

        self.fc_d = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, y_dim),
        )

    def forward(self, z):
        y = self.fc_d(z)
        return {"probs": torch.sigmoid(y)}
    
class Decoder_Y2(Bernoulli):
    def __init__(self, z_dim=2, y_dim=1):
        super(Decoder_Y2, self).__init__(cond_var=["z"], var=["y2"])
        self.y_dim = y_dim

        self.fc_d = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, y_dim),
        )

    def forward(self, z):
        y = self.fc_d(z)
        return {"probs":  torch.sigmoid(y)}
    

#### 分類器

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
            nn.Linear(1024, 2),
        )

    def forward(self, x):
        x = self.conv_e(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc_e(x)
        y = torch.sigmoid(x)
        return y

    
#### 比較 CVAE


class CVAE_Encoder(Normal):
    def __init__(self, z_dim=2, y_dim=2):
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
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8 ⇒ 4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),            
        )
        self.fc_y = nn.Sequential(
            nn.Linear(y_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),                   
        )      
       
        self.fc = nn.Sequential(
            nn.Linear(1024+256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, z_dim*2),
        )

    def forward(self, x, y):
        x = self.conv_e(x)
        x = x.view(-1, 64 * 4 *4)
        y = self.fc_y(y)
        x = torch.cat([x, y], dim=1)
        x = self.fc(x)
        mu = x[:, :self.z_dim]
        scale = F.softplus(x[:, self.z_dim:])
        return {"loc": mu, "scale": scale}


class CVAE_Decoder(Bernoulli):
    def __init__(self, z_dim=2, y_dim=2):
        super(CVAE_Decoder, self).__init__(cond_var=["z", "y"], var=["x"])

        # decode
        self.fc_y = nn.Sequential(
            nn.Linear(y_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),            
        )
        self.fc_z = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),                 
        )

        self.fc_d = nn.Sequential(
            nn.Linear(256+256, 1024),
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
        y = self.fc_y(y)
        z = self.fc_z(z)
        z = torch.cat([y, z], dim=1)
        z = self.fc_d(z)
        z = z.view(-1, 128, 8, 8)
        return {"probs": self.conv_d(z)}

