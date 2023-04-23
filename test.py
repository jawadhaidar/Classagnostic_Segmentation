from tqdm import tqdm 
import time

import torch

# Create two tensors
T1 = torch.Tensor([[0.6,0.6,0.7],[0.5,0.5,0.1]])
T2 = torch.Tensor([[1,1,1],[0,0,0]])

def naive_accuracy(predicted_mask,gt_mask):
    #compare to matrices
    predicted_mask[predicted_mask>0.5]=1
    predicted_mask[predicted_mask<=0.5]=0
    
    temp=predicted_mask==gt_mask
    total=(predicted_mask.shape[-1]*predicted_mask.shape[-2]) 
    acc= temp.sum()/total * 100
    return acc


