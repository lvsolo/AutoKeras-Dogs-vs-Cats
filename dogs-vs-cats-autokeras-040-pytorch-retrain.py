import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_size = (64,64)
import os
import cv2
import numpy as np


import autokeras as ak
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel,self).__init__()
        #load the model pretrained from autokeras
        self.origin_model = torch.load(MODEL_DIR)
        self.softmax = nn.Softmax()
    def forward(self,x):
        x = self.origin_model(x) 
        x = self.softmax(x)
        return x
MODEL_DIR = 'model.h5'
test_model = TestModel()
test_model.train()
test_model = nn.DataParallel(test_model, device_ids=[0]).cuda()
torch.save(test_model,'test_model')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(test_model.parameters(), lr=0.001, momentum=0.9)
index = [0,2,3,1]
train_whole_images = np.load('train_whole_images_'+str(image_size[0])+'.npy')
train_whole_labels = np.load('train_whole_labels_'+str(image_size[0])+'.npy')
train_whole_images = np.swapaxes(train_whole_images, 1, 3)
train_whole_images = np.swapaxes(train_whole_images, 2, 3)
batchsize = 128
for epoch in range(100):  # loop over the dataset multiple times
    for batch_ind in range(int(len(train_whole_images)/batchsize)):
        running_loss = 0.0
        i = batch_ind
        train_whole_images_batch = train_whole_images[i*batchsize:(i+1)*batchsize-1]
        train_whole_labels_batch = train_whole_labels[i*batchsize:(i+1)*batchsize-1]
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        inputs = torch.Tensor(train_whole_images_batch).cuda()
        outputs = test_model(inputs)
        loss = criterion(outputs.squeeze(), torch.Tensor(np.asarray(train_whole_labels_batch)).long().squeeze().cuda())
        loss.backward()
        optimizer.step()
        if i % (20) == 0:
            print('epoch:',epoch+1,',batch_ind/total_batch:',float(i/float(len(train_whole_images))*batchsize)*100,'%')
        
    torch.save(test_model,'test_model')
    with torch.no_grad():#this means the codes below is not trainable, grad dont need loaded to the gpu cache
        test_num = 100
        right = 0
        cat = 0
        dog = 0
        rand_ind = np.random.randint(0,len(train_whole_images)-1,test_num)
        predict = test_model(torch.Tensor(train_whole_images[rand_ind])).cpu()
        print(torch.argmax(predict,dim=1))
        label = train_whole_labels[rand_ind]
        print('label:', label)
        dog = np.sum(label)
        cat = test_num - np.sum(label)
        right = np.sum(torch.argmax(predict, dim=1).numpy()==label)
        print(torch.argmax(predict, dim=1).numpy()==label)
        print(epoch+1,':', float(right)/test_num)
        print('cat:', cat)
        print('dog:', dog)
        #clear the cuda cache
        torch.cuda.empty_cache()
print('Finished Training')
