import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


#print(modeledited)


class Usiam(nn.Module):
    def __init__(self):
        super(Usiam,self).__init__()
        self.num_maps=8
        self.in_channels=3
        self.model1 = smp.Unet(
                encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=self.in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.num_maps,                      # model output channels (number of classes in your dataset)
            ) #freeze is not working

        #freeze the backbone param
        for id,child in enumerate(self.model1.children()):
            if id==0:
                for param in child.parameters():
                    param.requires_grad=False
                    #print(param.requires_grad)

        self.batch_norm=nn.BatchNorm2d(self.num_maps)
        self.relu=nn.ReLU()
        self.conv=nn.Conv2d(in_channels=self.num_maps + self.in_channels, out_channels=3, kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.conv_last=nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.sigmoid=nn.Sigmoid()

    def forward(self,real,inpaint):
        #out=self.model_backbone(x)
        
        real_out=self.relu(self.model1(real))
        inpaint_out=self.relu(self.model1(inpaint)) # should I add batch norm?
        real_out=torch.cat((real_out, real), 1)
        inpaint_out=torch.cat((inpaint_out, inpaint), 1)
        #concatinate (you can add as well real and inpaint)
        combined= torch.pow(real_out - inpaint_out,2) # this enforces a prior
        out=self.sigmoid(self.conv_last(self.conv(combined)))
        '''
        real_out=self.relu(self.model1(real))
        inpaint_out=self.relu(self.model1(inpaint)) # should I add batch norm?
        real_out=torch.cat((real_out, real), 1)
        inpaint_out=torch.cat((inpaint_out, inpaint), 1)
        #concatinate (you can add as well real and inpaint)
        combined= torch.pow(real_out - inpaint_out,2)
        out=self.sigmoid(torch.sum(combined,1)).unsqueeze(1)
        '''

        return out


if __name__=="__main__":

    mymodel=Usiam()
    inp=torch.rand(1,3,512,512)
    print(mymodel(inp,inp).shape)