import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import partB_AE
from partB_test import *

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib.pyplot as plt
dataset_path = "visualization.npy"
dataset = load_data(dataset_path)
model_path = "model_ae_best.pkl"
y = [0 if i < 2500 else 1 for i in range(5000)]
def plot_scatter(feat, label, savefig=None):
    X = feat[:, 0]
    Y = feat[:, 1]
    plt.scatter(X, Y, c = label)
    plt.legend(loc='best')
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    return
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = partB_AE.Autoencoder_1()
model.load_state_dict(torch.load(model_path , map_location = device))
test_y, embedded_x = cluster(dataset , model , device)
plot_scatter(embedded_x, y, savefig = "scatter.png")

#plot original image
plt.figure(figsize=(10,4))
indexes = [1,2,3,4,5,6]
trainX = np.load(dataset_path)
imgs = trainX[indexes,]
for i, img in enumerate(imgs):
    plt.subplot(2, 6, i+1, xticks=[], yticks=[])
    plt.imshow(img)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = partB_AE.Autoencoder_1()
model.load_state_dict(torch.load(model_path , map_location = device))
loaded_data = load_data(dataset_path)
inp = loaded_data

latents, recs = model(inp[indexes,])
recsNumpy = ((recs.cpu()+1)/2).detach().numpy()
recsT = np.transpose(recsNumpy, (0, 2, 3, 1))
for i, img in enumerate(recsT):
    plt.subplot(2, 6, 6+i+1, xticks=[], yticks=[])
    plt.imshow(img)
  
plt.tight_layout()