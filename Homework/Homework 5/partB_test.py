#test.py
import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import partB_AE
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
dataset_path = "trainX.npy"
submission = "submission.csv"
model_path = "model_ae_best.pkl"
def load_data(path):
	test_x = np.load(path)
	test_x = np.transpose(test_x , (0 , 3 , 1 , 2))
	test_x = test_x / 255
	test_x = 2 * test_x - 1
	return torch.FloatTensor(test_x)

def cluster(test_x , model , device):
	# Hyper-parameter.
	batch_size = 256

	test_loader = DataLoader(test_x , batch_size = batch_size , shuffle = False)

	model.to(device)
	model.eval()
	new_x = list()
	with torch.no_grad():
		for data in test_loader:
			data = data.to(device)
			(encode , decode) = model(data)
			new_x.append(encode.cpu().detach().numpy())
	new_x = np.concatenate(new_x , axis = 0)
	new_x = new_x.reshape((test_x.shape[0] , -1))
	new_x = (new_x - np.mean(new_x , axis = 0)) / np.std(new_x , axis = 0)
	PCA = KernelPCA(n_components = 100, kernel = 'rbf', n_jobs = -1)
	new_x = PCA.fit_transform(new_x)
	tsne = TSNE(n_components = 2)
	x_embedded = tsne.fit_transform(new_x)

	kmeans = KMeans(n_clusters = 2)
	kmeans.fit(x_embedded)
	test_y = kmeans.labels_
	return test_y, x_embedded

def output(test_y):
	number_of_data = test_y.shape[0]
	df = pd.DataFrame({'id' : np.arange(number_of_data) , 'label' : test_y})
	df.to_csv(submission , index = False)
	return


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_x = load_data(dataset_path)

model = partB_AE.Autoencoder_1()
model.load_state_dict(torch.load(model_path , map_location = device))
train_y, _ = cluster(train_x , model , device)
output(train_y)