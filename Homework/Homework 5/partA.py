#ML2021 HW5 Part A
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import time


file_path = "Aberdeen"
data_list = [os.path.join(file_path, img) for img in os.listdir(file_path)]
output_path = ""
image_size = 128

def reshape_array(arr):
    arr = np.array(arr)
    if len(arr.shape) == 1: 
        arr = np.reshape(arr, (1, arr.shape[0]))
    return arr


def reshape_image(img):
    global image_size
    img -= np.min(img)
    img /= np.max(img)
    img = (img * 255).astype(np.uint8)
    img = np.reshape(img, (image_size, image_size, 3))
    return img

imagelist = []
for image in data_list:
    image = image.replace("/._","/")
    if ".DS_Store" in image:
        continue
    arrayImage = io.imread(image, plugin='matplotlib')
    arrayResizeImage = transform.resize(arrayImage, (image_size, image_size, 3))
    imagelist.append(arrayResizeImage.flatten())

n = 5

imagelist = reshape_array(imagelist) 
avg = reshape_array(np.mean(imagelist, axis=0)) 
U, S, V = np.linalg.svd((imagelist - avg).T, full_matrices=False)

S = S[0:n]
U = U[:, 0:n]

random_index = [132, 223, 101, 20, 45]
random_list = [imagelist[i] for i in random_index]
random_list = reshape_array(random_list)
random_list -= avg
Z = np.dot(random_list, U)
Xhat = np.dot(Z, U.T)

ReconImage = Xhat + avg

# output average face
io.imsave(os.path.join(output_path, "Average_face.png"), avg.reshape(image_size, image_size, 3))

# output top 5 eigenface
for i in range(5):
    io.imsave(os.path.join(output_path, "Top_{}_EigenFace.png".format(i)), reshape_image(U[:, i]))

# output 5 random reconstructed images
for i in range(len(random_index)):
    io.imsave(os.path.join(output_path, "Reconstruct_img{}.png".format(random_index[i])), reshape_image(ReconImage[i]))