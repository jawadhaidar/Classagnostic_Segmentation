
from __init__ import*
from torchsummary import summary
import torch

# get data 
mytransform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((288,512)),
        transforms.ConvertImageDtype(torch.float),
    ])

dataset_one=Data_Usiam(mytransform)

#data loader 

val_loader=DataLoader(dataset=dataset_one,batch_size=1,shuffle=True) # val or test 

# model 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'the device is {DEVICE}')
model=Usiam().to(device=DEVICE)

#inference 
acc=inference(model_path=r'Classagnostic_Segmentation\saved_model\model_large.pth',val_loader=val_loader,DEVICE=DEVICE,draw=True)
print(acc)
