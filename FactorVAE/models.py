from pixyz.distributions import Normal, Bernoulli, Deterministic
import torch
import torch.nn as nn
from torch.nn import functional as F

class Encoder(Normal):
    def __init__(self, z_dim=10):
        super(Encoder, self).__init__(cond_var=["x"], var=["z"])

        self.z_dim = z_dim
        # encode
        self.conv_e = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),    # 64 ⇒ 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 ⇒ 16
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
            nn.Linear(1024, 2*self.z_dim),
        )

    def forward(self, x):
        x = self.conv_e(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc_e(x)
        mu = x[:, :self.z_dim]
        scale = F.softplus(x[:, self.z_dim:])
        return {"loc": mu, "scale": scale}
        
        
class Decoder(Bernoulli):
    def __init__(self, z_dim=10):
        super(Decoder, self).__init__(cond_var=["z"], var=["x"])

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
    
class Discriminator(Deterministic):
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__(cond_var=["z"], var=["t"], name="d")
        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        t = self.model(z)
        return {"t": t}
    