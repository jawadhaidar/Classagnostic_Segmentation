import torch 
import torchvision 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math 
import cv2 as cv
#from skyimage import io
from PIL import Image
import os 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Data_Usiam(Dataset):
    def __init__(self,transform=None):
        #should you add the super constructer?
        #floders path
        self.real_path=r'Classagnostic_Segmentation\data\real_imgs'
        self.inpaint_path=r'Classagnostic_Segmentation\data\inpainted_imgs' # two options here
        self.mask_path=r'Classagnostic_Segmentation\data\masks'

        #image name list 
        self.image_real_name_list=os.listdir(self.real_path)
        self.image_inpaint_name_list=os.listdir(self.inpaint_path)
        self.image_mask_name_list=os.listdir(self.mask_path)
        #transform
        self.transform=transform
       
        #self number of samples
        self.num_samples=len(self.image_real_name_list)
        

    def __getitem__(self, index):
        #get image real
        img_real=cv.cvtColor(cv.imread(os.path.join(self.real_path,self.image_real_name_list[index])), cv.COLOR_BGR2RGB) 
        #get image inpaint
        img_inpaint=cv.cvtColor(cv.imread(os.path.join(self.inpaint_path,self.image_inpaint_name_list[index])), cv.COLOR_BGR2RGB)
        #get label mask
        img_mask=cv.cvtColor(cv.imread(os.path.join(self.mask_path,self.image_mask_name_list[index])), cv.COLOR_BGR2RGB)
        #binarize the mask
        img_mask=img_mask[:,:,[1]]/255

        #transform
        if self.transform:
            
            img_real=self.transform(img_real)
            img_inpaint=self.transform(img_inpaint)
            img_mask=self.transform(img_mask)[:,:]
            
            #img_real,img_inpaint,img_mask=self.transform(img_real,img_inpaint,img_mask)

        return (img_real,img_inpaint,img_mask)

    def __len__(self):
        return self.num_samples


       

# common transform compose
## resize 
## change to tensor 

if __name__=='__main__':
    mytransform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((288,512)),
                transforms.ConvertImageDtype(torch.float),
            ])

    dataset_one=Data_Usiam(mytransform)
    #train_set, val_set = torch.utils.data.random_split(dataset_one, [dataset_one.num_samples - 10, 10])
    #data loader 
    train_loader=DataLoader(dataset=dataset_one,batch_size=8,shuffle=True)
    #val_loader=DataLoader(dataset=val_set,batch_size=8,shuffle=True)

    for imgs,inpainted,labels in train_loader:
        print(imgs.shape)
        print(inpainted.shape)
        print(labels.shape)
        for ind in range(imgs.shape[0]):
            a=inpainted[ind,0,:,:].squeeze().detach().to('cpu').numpy()
            plt.subplot(1,3,1)
            plt.title("inpainted")
            plt.imshow(a)
            plt.subplot(1,3,2)
            plt.title("ground truth")
            plt.imshow(labels[ind,0,:,:].squeeze().detach().to('cpu').numpy())
            plt.subplot(1,3,3)
            plt.title("img")
            plt.imshow(imgs[ind,0,:,:].squeeze().detach().to('cpu').numpy())
            plt.show()

    print(dataset_one[0][1].shape)

 