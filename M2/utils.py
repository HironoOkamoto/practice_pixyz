import matplotlib.pyplot as plt
import torch
import umap
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

       
def plot_sample(D, epoch, z_dim=63, y_dim=10, sample_num=100):
    # fixed noise & condition サンプル画像を作る用
    np.random.seed(42)
    sample_z_ = torch.zeros((sample_num, z_dim))
    for i in range(10):
        sample_z_[i*y_dim] = torch.from_numpy(np.random.randn(1, z_dim))#torch.rand(1, z_dim) # 正規分布randn or 一様分布rand
        for j in range(1, y_dim):
            sample_z_[i*y_dim + j] = sample_z_[i*y_dim]

    temp = torch.zeros((10, 1))
    for i in range(y_dim):
        temp[i, 0] = i

    temp_y = torch.zeros((sample_num, 1))
    for i in range(10):
        temp_y[i*y_dim: (i+1)*y_dim] = temp

    sample_y_ = torch.zeros((sample_num, y_dim))
    sample_y_.scatter_(1, temp_y.type(torch.LongTensor), 1)
    sample_z_, sample_y_ = sample_z_.to(device), sample_y_.to(device)

    samples = D(sample_z_, sample_y_)
    samples = samples["probs"].cpu().data.numpy().squeeze()
    
    # 横軸 z固定, y変化
    # 縦軸 z変化, y固定
    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(samples[i], cmap=plt.cm.gray)
    #plt.savefig("./logs/generate.png")
    fig.suptitle("epoch: {}".format(epoch+1), va="bottom", fontsize=30, y=0.9)
    plt.savefig("./logs/mnist_gif/{}.png".format(epoch+1), bbox_inches="tight", pad_inches=0.0)
    plt.show()
    
    
def plot_loss(train_hist):
#     plt.figure(figsize=(10, 10))
#     plt.subplot(222)
#     plt.title("Encoder loss")
#     plt.plot(train_hist["E_loss"])
#     plt.subplot(223)
#     plt.title("Decoder loss")
#     plt.plot(train_hist["D_loss"])
#     plt.subplot(224)
#     plt.title("precision: {:.4f}".format(train_hist["precision"][-1]))
    #plt.ylim(0.95, 0.995)
    plt.plot(train_hist["precision"])
    plt.savefig("./logs/plot_loss.png", bbox_inches="tight", pad_inches=0.0)
    plt.show()

    
# test
def compute_precision(C, test_loader):
    C.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            pred = C(x)
            preds.extend(pred["probs"].detach().cpu().numpy())
            ys.extend(y)
    preds = np.array(preds)
    ys = np.array(ys)
            
    precision = precision_score(ys, preds.argmax(1), average="macro")
    print(precision)
    print(confusion_matrix(ys, preds.argmax(1)))
    print(classification_report(ys, preds.argmax(1)))
    return precision
            
    
