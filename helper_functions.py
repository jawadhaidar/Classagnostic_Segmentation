from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

def train(epochs,train_loader,model,loss_fn,optimizer,DEVICE):

    for epoch in tqdm(range(epochs)):
        running_loss=0
        for i, batched_sample in enumerate(train_loader):

            real_img,inpainted_img,label=batched_sample
            #forward pass
            out=model(real_img.to(device=DEVICE),inpainted_img.to(device=DEVICE)).float()
            #calculate loss
            loss = loss_fn(out, label.to(device=DEVICE))
            #print(f'max of out {out.max()} and maxof label {label.max()}')
            #plt.imshow(out[0,0,:,:].squeeze().detach().numpy())
            #plt.show()
            #backward pass
            loss.backward()
            #update weights
            optimizer.step()
            #zero gradients
            optimizer.zero_grad()
            #print batch number out of total 
            #print epoch number
            # Gather data and report
            running_loss += loss.item()
            print(f'batch {i} out of {len(train_loader)} with batch loss {loss.item():.20f} and acc : {naive_accuracy(out,label)}')

        print(f'epoch {epoch} || loss : {running_loss}')
        torch.save(model, r'Classagnostic_Segmentation\saved_model\model_large.pth') #to be changed

        

def inference(model_path,val_loader,DEVICE,draw):
    model=torch.load(model_path).to(device=DEVICE)
    acc_sum=0
    model.eval()
    for i, batched_sample in enumerate(val_loader):
        real_img,inpainted_img,label=batched_sample
        label=label.to(device=DEVICE)
        real_img=real_img.to(device=DEVICE)
        inpainted_img=inpainted_img.to(device=DEVICE)

        #forward passss
        out=model(real_img,inpainted_img).float()
        
        acc_batch=naive_accuracy(out,label)
        print(acc_batch)
        #draw 
        if draw:
            for ind in range(out.shape[0]):
                a=out[ind,0,:,:].squeeze().detach().to('cpu').numpy()
                plt.subplot(1,2,1)
                plt.title("predicted")
                plt.imshow(a)
                plt.subplot(1,2,2)
                plt.title("ground truth")
                plt.imshow(label[ind,0,:,:].squeeze().detach().to('cpu').numpy())
                #plt.savefig(os.path.join(f'Classagnostic_Segmentation\results\validation_withdilmask',"{i}.png"))
                plt.show()

        acc_sum+=acc_batch

    acc_total=acc_sum/ i+1
    return acc_total

def naive_accuracy(predicted_mask,gt_mask):
    #compare to matrices
    predicted_mask[predicted_mask>0.5]=1
    predicted_mask[predicted_mask<=0.5]=0
    temp=predicted_mask==gt_mask
    batch_size=predicted_mask.shape[0]
    total=(predicted_mask.shape[-1]*predicted_mask.shape[-2]) * batch_size
    acc= temp.sum()/total * 100
    return acc

def custom_loss(predicted,groundtruth,bbx,slarge,smeduim,ssmall):

    #extract from bbx 
    c=10 #to be changed
    v=10
    xmin,ymin,xmax,ymax=bbx[0],bbx[1],bbx[2],bbx[3]
    weights=torch.zeros_like(predicted)

    #easy
    weights[:,:]=slarge
    weights[ymin:ymax,xmin:xmax]=smeduim 
    weights[ymin+c:ymax-c,xmin+v:xmax-v]=ssmall 
    

    loss= torch.sum(-groundtruth*torch.log(predicted)*weights)

    return loss 
 

if __name__=="__main__":
    predicted=torch.ones((100,100))
    groundtruth=torch.zeros((100,100))
    bbx=[20,10,80,50]
    slarge=1
    smeduim=1
    ssmall=1

    loss=custom_loss(predicted,groundtruth,bbx,slarge,smeduim,ssmall)
    print(loss)
