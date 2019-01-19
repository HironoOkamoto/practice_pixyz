import matplotlib.pyplot as plt
import torch
import umap
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_sample(G, epoch, z_dim=100, sample_num=100):
    # fixed noise & condition サンプル画像を作る用
    np.random.seed(42)
    sample_z_ = torch.from_numpy(np.random.randn(sample_num, z_dim).astype("float32")).to(device)
    samples = G(sample_z_)["x"]
    samples = samples.cpu().data.numpy().squeeze()
    
    # 横軸 z固定, y変化
    # 縦軸 z変化, y固定
    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(samples[i], cmap=plt.cm.gray)
    fig.suptitle("epoch: {}".format(epoch+1), va="bottom", fontsize=30, y=0.9)
    plt.savefig("./logs/mnist_gif/{:0>3}.png".format(epoch+1), bbox_inches="tight", pad_inches=0.0)
    plt.show()
    
