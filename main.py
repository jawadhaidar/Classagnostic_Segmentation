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
#train_set, val_set = torch.utils.data.random_split(dataset_one, [dataset_one.num_samples - 10, 10])

#data loader 
train_loader=DataLoader(dataset=dataset_one,batch_size=32,shuffle=True)
#val_loader=DataLoader(dataset=val_set,batch_size=1,shuffle=True)

# model 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'the device is {DEVICE}')
model=Usiam().to(device=DEVICE)


# freeze the backbone

'''
for child in model.children():
    for id,ch in enumerate(child.children()):
        if id==0:
            for param in ch.parameters():
                #param.requires_grad = False
                print(param.requires_grad )


print(model)

for child in model_ft.children()[0]:
    for param in child.parameters():
        param.requires_grad = False
'''

#criterion
#loss_fn=nn.BCELoss() #nn.CrossEntropyLoss()
loss_fn=smp.losses.DiceLoss(mode='binary')  
#optimizer 
optimizer=torch.optim.Adam(model.parameters(), lr=0.1, momentum=0.9) #lr was 0.001

#train
epochs=20
train(epochs,train_loader,model,loss_fn,optimizer,DEVICE)

#
#summary(model, [(3,288,512), (3,288,512)])


#acc=inference(model_path=r'Classagnostic_Segmentation\saved_model\model.pth',val_loader=val_loader,DEVICE=DEVICE,draw=True)
#print(acc)

'''
list_p=list(val_loader)

model = torch.load(r'Classagnostic_Segmentation\saved_model\model.pth')
model.eval()

x=2
output=model(list_p[x][0],list_p[x][1])
print(output.shape)
plt.imshow(output[0,0,:,:].squeeze().detach().numpy())
plt.show()
lab=list_p[x][2]
plt.imshow(lab[0,0,:,:].squeeze().detach().numpy())
plt.show()
'''