import os 

image_size = (64,64)
import os
import cv2
import numpy as np
#load numpy data file
train_images_npy = 'train_whole_images_'+str(image_size[0])+'.npy'
train_labels_npy = 'train_whole_labels_'+str(image_size[0])+'.npy'
train_whole_images = np.load(train_images_npy)
train_whole_labels = np.load(train_labels_npy)

import autokeras as ak
import torch
import torchvision
classifier = ak.ImageClassifier(verbose = True, path = 'autokeras_temp_train')
classifier.fit(x=np.asarray(train_whole_images, dtype = np.uint8), y=np.asarray(train_whole_labels,dtype =np.float16),time_limit = 60*50*1)
# the file name to store the model both structure and parameters
MODEL_DIR = 'model.h5'
# different methods to save the model, some of which may differ from different vision.
# 1)
#classifier.export_keras_model(MODEL_DIR)
# 2）
#from autokeras.utils import pickle_to_file,pickle_from_file
#pickle_to_file(classifier,MODEL_DIR) 
# 3）
#torch.save(classifier.cnn.best_model.produce_model(),MODEL_DIR)
model = torch.load(MODEL_DIR)
model.eval()
with torch.no_grad():
    right = 0
    cat = 0
    dog = 0
    # the number of samples chosen to test the model
    test_num = 100
    rand_ind = np.random.randint(0,len(train_whole_images)-1,test_num)
    predict = model(torch.Tensor(train_whole_images[rand_ind])).cpu()
    label = train_whole_labels[rand_ind]
    print(torch.argmax(predict,dim=1))
    print('label:', label)
    print(torch.argmax(predict, dim=1).numpy()==label)
    dog = np.sum(label)
    cat = 100 - np.sum(label)
    right = np.sum(torch.argmax(predict, dim=1).numpy()==label)
    print('accuracy:', float(right)/100)
    print('cat:', cat)
    print('dog:', dog)
