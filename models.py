import torch.nn as nn
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DFN(nn.Module):
    def __init__(self, z_dim, label_dim=10):
        super(DFN, self).__init__()
        self.z_dim = z_dim
        self.U = nn.Embedding(z_dim, label_dim)
        self.U.weight.data.uniform_(0, 0.05)
        self.U_ = None

    def forward(self, z):
        self.U_ = self.U(torch.from_numpy(np.arange(self.z_dim)).to(device))
        r = torch.matmul(z, self.U_)
        return r

class Encoder_MNIST(nn.Module):
    def __init__(self, z_dim):
        super(Encoder_MNIST, self).__init__()
        # encode
        self.conv_e = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),    # 28 ⇒ 14
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14 ⇒ 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc_e = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, z_dim),
        )

    def forward(self, x):
        x = self.conv_e(x)
        x = x.view(-1, 128*7*7)
        x = self.fc_e(x)
        return x

class Decoder_MNIST(nn.Module):
    def __init__(self, z_dim):
        super(Decoder_MNIST, self).__init__()
        # decode
        self.fc_d = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128*7*7),
            nn.LeakyReLU(0.2)
        )
        self.conv_d = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc_d(z)
        h = h.view(-1, 128, 7, 7)
        return self.conv_d(h)
    
class Encoder_SVHN(nn.Module):
    def __init__(self, z_dim):
        super(Encoder_SVHN, self).__init__()
        # encode
        self.conv_e = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),    # 32 ⇒ 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16 ⇒ 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc_e = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, z_dim),
        )

    def forward(self, x):
        x = self.conv_e(x)
        x = x.view(-1, 128*8*8)
        x = self.fc_e(x)
        return x

class Decoder_SVHN(nn.Module):
    def __init__(self, z_dim):
        super(Decoder_SVHN, self).__init__()
        # decode
        self.fc_d = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128*8*8),
            nn.LeakyReLU(0.2)
        )
        self.conv_d = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc_d(z)
        h = h.view(-1, 128, 8, 8)
        return self.conv_d(h)
    
class Encoder_USPS(nn.Module):
    def __init__(self, z_dim):
        super(Encoder_USPS, self).__init__()
        # encode
        self.conv_e = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),    # 16 ⇒ 8
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 8 ⇒ 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc_e = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, z_dim),
        )

    def forward(self, x):
        x = self.conv_e(x)
        x = x.view(-1, 128*7*7)
        x = self.fc_e(x)
        return x

class Decoder_USPS(nn.Module):
    def __init__(self, z_dim):
        super(Decoder_USPS, self).__init__()
        # decode
        self.fc_d = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128*7*7),
            nn.LeakyReLU(0.2)
        )
        self.conv_d = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc_d(z)
        h = h.view(-1, 128, 7, 7)
        return self.conv_d(h)   


class Discriminator_z(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator_z, self).__init__()
        h_dim = 512
        self.fc = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(0.2),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(0.2),
            nn.Linear(h_dim, 1)
        )
        initialize_weights(self)

    def forward(self, x):
        x = self.fc(x)
        return nn.Sigmoid()(x)
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()