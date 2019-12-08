import os 
import cv2
import numpy as np
image_size = (64,64)
#your path of the datasets
data_dir = '/mnt/HDD/Datasets/kaggle/dogs-vs-cats/'
train_dir = data_dir + "train/"
test_dir = data_dir + 'test1/'
test_list = os.listdir(test_dir)
train_list = os.listdir(train_dir)
train_whole_images = []
train_whole_labels = []
test_whole_images = []
for ind,name in enumerate(test_list):
    print(ind)
    test_whole_images.append(cv2.resize(cv2.imread(test_dir+name),image_size))

for ind,name in enumerate(train_list):
    print(ind)
    train_whole_images.append(cv2.resize(cv2.imread(train_dir+name),image_size))
    if 'cat' in name:
        train_whole_labels.append(0)
    else:
        train_whole_labels.append(1)
# restore the datasets in numpy file
np.save('test_whole_images_'+str(image_size[0])+'.npy',np.asarray(test_whole_images))

np.save('train_whole_images_'+str(image_size[0])+'.npy',np.asarray(train_whole_images))
np.save('train_whole_labels_'+str(image_size[0])+'.npy',np.asarray(train_whole_labels))
# reload the dataset stored in numpy file
#test_whole_images = np.load('test_whole_images_'+str(image_size[0])+'.npy')
#test_whole_labels = np.load('test_whole_labels_'+str(image_size[0])+'.npy')
