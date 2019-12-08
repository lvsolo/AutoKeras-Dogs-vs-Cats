import os 
import cv2
import numpy as np

import autokeras as ak
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
# create a new model to change the last 1 layer in the origin model generated from autokeras
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel,self).__init__()
        # load origin model
        self.origin_model = torch.load(MODEL_DIR)
        self.softmax = nn.Softmax()
    def forward(self,x):
        x = self.origin_model(x) 
        x = self.softmax(x)

        return x

image_size = (64,64)
MODEL_DIR = 'test_model'

test_model = TestModel()
print(test_model)
# make the training done by gpu
test_model = nn.DataParallel(test_model, device_ids=[0]).cuda()
test_model.eval()
test_whole_images = np.swapaxes(test_whole_images, 1, 3)
test_whole_images = np.swapaxes(test_whole_images, 2, 3)
batchsize = 100
with torch.no_grad():
    for i in range(int(len(test_whole_images)/batchsize)):
        predict = test_model(torch.Tensor(test_whole_images[i*batchsize:(i+1)*batchsize])).cpu()
        predict = torch.argmax(predict,dim=1).numpy()
        print(predict)
        
        ids = test_whole_ids[i*batchsize:(i+1)*batchsize]
        with open('submission.csv', 'a+') as f:
            #f.write('id,' + ','.join(test_whole_ids[i]) + '\n')
            for i, output in zip(ids, predict):
                f.write(str(i) + ',' + ','.join(
                    str(output)) + '\n')
print('Finished Training')
