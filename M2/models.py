import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F

from pixyz.distributions import Normal, Bernoulli, RelaxedCategorical, Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# q(z|x, y)
class Encoder(Normal):
    def __init__(self, z_dim=63, y_dim=10):
        super(Encoder, self).__init__(cond_var=["x", "y"], var=["z"], name="q")

        self.z_dim = z_dim

        # encode
        self.conv_e = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 28 ⇒ 14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14 ⇒ 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 7 *7,  40),
        )      
        self.fc2 = nn.Sequential(
            nn.Linear(128 * 7 * 7,  y_dim),
        )        
        
        self.fc = nn.Sequential(
            nn.Linear(40+y_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*self.z_dim),
        )

    def forward(self, x, y):
        x = self.conv_e(x)
        x = x.view(-1, 128 * 7 * 7)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = torch.cat([x1, x2*y], dim=1)
        x = self.fc(x)
        mu = x[:, :self.z_dim]
        scale = F.softplus(x[:, self.z_dim:])
        return {"loc": mu, "scale": scale}

# p(x|z, y)
class Decoder(Bernoulli):
    def __init__(self, z_dim=63, y_dim=10):
        super(Decoder, self).__init__(cond_var=["z", "y"], var=["x"])
        
        self.z_dim = z_dim 

        # decode
        self.fc1 = nn.Sequential(
            nn.Linear(self.z_dim, 40),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.z_dim, y_dim),
        )
        
        self.fc_d = nn.Sequential(
            nn.Linear(40+y_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128 * 7 * 7),
            nn.LeakyReLU(0.2)
        )
        self.conv_d = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z, y):
        z1 = self.fc1(z)
        z2 = self.fc2(z)
        z = torch.cat([z1, z2*y], dim=1)
        h = self.fc_d(z)
        h = h.view(-1, 128, 7, 7)
        return {"probs": self.conv_d(h)}
    
# q(z|x)
class Encoder_m1(Normal):
    def __init__(self, z_dim=63):
        super(Encoder_m1, self).__init__(cond_var=["x"], var=["z"], name="q")

        self.z_dim = z_dim

        # encode
        self.conv_e = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 28 ⇒ 14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14 ⇒ 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*self.z_dim),
        )

    def forward(self, x):
        x = self.conv_e(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc(x)
        mu = x[:, :self.z_dim]
        scale = F.softplus(x[:, self.z_dim:])
        return {"loc": mu, "scale": scale}

# p(x|z)
class Decoder_m1(Bernoulli):
    def __init__(self, z_dim=63):
        super(Decoder_m1, self).__init__(cond_var=["z"], var=["x"])
        
        self.z_dim = z_dim 

        # decode
        self.fc_d = nn.Sequential(
            nn.Linear(self.z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128 * 7 * 7),
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
        return {"probs": self.conv_d(h)}
    
    
# classifier q(y|x)
class Classifier(RelaxedCategorical):    
    def __init__(self, y_dim=10):
        super(Classifier, self).__init__(cond_var=["x"], var=["y"], temperature=0.5, name="q")
        self.input_height = 28
        self.input_width = 28

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, padding=2),   # 28x28 ⇒ 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, padding=2), # 14x14 ⇒ 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc =  nn.Sequential(
            nn.Linear((self.input_height // 4) * (self.input_width // 4) * 128, 256),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(256, y_dim),
            nn.Dropout(p=0.4),
            nn.Softmax(dim=1)            
        )
        initialize_weights(self)
        
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c2_flat = c2.view(c2.size(0), -1)
        out = self.fc(c2_flat)
        return {"probs": out}
    
# latent classifier q(y|z)
class LatentClassifier(RelaxedCategorical):    
    def __init__(self, z_dim=63, y_dim=10):
        super(LatentClassifier, self).__init__(cond_var=["z"], var=["y"], temperature=0.5, name="q")
        self.fc =  nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(256, y_dim),
            nn.Dropout(p=0.4),
            nn.Softmax(dim=1)            
        )
        initialize_weights(self)
        
    def forward(self, z):
        out = self.fc(z)
        return {"probs": out}
    
# q(z_2|z_1, y)
class LatentEncoder(Normal):
    def __init__(self, z_dim=63, z2_dim=20, y_dim=10):
        super(LatentEncoder, self).__init__(cond_var=["z", "y"], var=["z2"], name="q")
        self.z_dim = z_dim
        self.z2_dim = z2_dim

        self.fc1 = nn.Sequential(
            nn.Linear(z_dim,  40),
        )      
        self.fc2 = nn.Sequential(
            nn.Linear(z_dim,  y_dim),
        )                
        self.fc = nn.Sequential(
            nn.Linear(40+y_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*z2_dim),
        )

    def forward(self, z, y):
        z_1 = self.fc1(z)
        z_2 = self.fc2(z)
        z = torch.cat([z_1, z_2*y], dim=1)
        z = self.fc(z)
        mu = z[:, :self.z2_dim]
        scale = F.softplus(z[:, self.z2_dim:])
        return {"loc": mu, "scale": scale}
    
# p(z_1|z_2, y)
class LatentDecoder(Normal):
    def __init__(self, z_dim=63, z2_dim=20, y_dim=10):
        super(LatentDecoder, self).__init__(cond_var=["z2", "y"], var=["z"])
        
        self.z_dim = z_dim 
        self.z2_dim = z2_dim 

        # decode
        self.fc1 = nn.Sequential(
            nn.Linear(z2_dim, 40),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(z2_dim, y_dim),
        )
        
        self.fc_d = nn.Sequential(
            nn.Linear(40+y_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, z_dim*2),
        )
        
    def forward(self, z2, y):
        z_1 = self.fc1(z2)
        z_2 = self.fc2(z2)
        z = torch.cat([z_1, z_2*y], dim=1)
        z = self.fc_d(z)
        mu = z[:, :self.z2_dim]
        scale = F.softplus(z[:, self.z2_dim:])
        return {"loc": mu, "scale": scale}
    
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
    