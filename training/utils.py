import matplotlib.pyplot as plt
import torch
import umap
import numpy as np
from torch import nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def latent_space_plot(data_loader, E, title):
    E.eval()
    zs = []
    ys = []
    for batch_idx, (x, y) in enumerate(data_loader):
        z = E(x.to(device))
        ys.extend(y)
        zs.extend(z.cpu().detach().numpy())
    zs = np.array(zs)
    ys = np.array(ys)

    embedding = umap.UMAP(random_state=42).fit_transform(zs)
    #embedding = TSNE(random_state=42).fit_transform(zs)
    plt.style.use("ggplot")
    plt.figure(figsize=(6, 5))
    plt.title(title)
    plt.scatter(embedding[:,0],embedding[:,1], c=ys, cmap=plt.cm.rainbow, s=2)
    plt.colorbar()
    plt.show()

    
def encoder_plot(data_loader, E, D):
    E.eval()
    D.eval()
    
    images, labels = iter(data_loader).next()
    
    images = images.to(device)
    z = E(images)
    samples = D(z)
    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()
    
    print("↓generate")
    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.imshow(samples[i], cmap=plt.cm.gray)
    images = images.cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()
    for i in range(10):
        plt.subplot(2, 10, i+11)
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.imshow(images[i], cmap=plt.cm.gray)
    plt.show()
    print("↑true")
    
    
# solver 
def training_AE(train_loader, test_loader, E, D, E_optimizer, D_optimizer, criterion, epoch_num):
    recon_loss_list = []
    for k in range(epoch_num):
        for batch_idx, (x, y) in tqdm(enumerate(train_loader)):
            E.train()
            D.train()

            x = x.to(device)

            # 再構成
            E_optimizer.zero_grad()
            D_optimizer.zero_grad()

            #1. sample from q(z|x) 
            x_ = D(E(x))
            recon_loss = criterion(x_, x)
            recon_loss.backward()

            E_optimizer.step()
            D_optimizer.step()

            recon_loss_list.append(recon_loss.detach())

            #if (batch_idx + 1) % 400 == 0:
    encoder_plot(test_loader, E, D)
    plt.plot(recon_loss_list)
    plt.show()
      
    
def training_DAE(train_loader, test_loader, E, D, dfn, E_optimizer, D_optimizer, dfn_optimizer, criterion, epoch_num):
    recon_loss_list = []
    rate_loss_list = []
    for k in range(epoch_num):
        for batch_idx, (x, y) in tqdm(enumerate(train_loader)):
            E.train()
            D.train()

            x = x.to(device)
            y = torch.eye(10)[y].to(device)

            # 再構成
            E_optimizer.zero_grad()
            D_optimizer.zero_grad()

            x_ = D(E(x))
            recon_loss = criterion(x_, x)
            recon_loss.backward()

            E_optimizer.step()
            D_optimizer.step()

            # 評価予測
            E_optimizer.zero_grad()
            dfn_optimizer.zero_grad()

            z = E(x)
            r = dfn(z)
            rate_loss = criterion(r, y) 
            rate_loss.backward()

            E_optimizer.step()
            dfn_optimizer.step()

            recon_loss_list.append(recon_loss.detach())
            rate_loss_list.append(rate_loss.detach())
            
            #if (batch_idx + 1) % 400 == 0:
            
    encoder_plot(test_loader, E, D)
    plt.plot(recon_loss_list)
    plt.show()
    plt.plot(rate_loss_list)
    plt.show()

             

                
                