import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchsummary import summary

#print(modeledited)

class UsiamSimple(nn.Module):
    def __init__(self):
        super(UsiamSimple,self).__init__()
        self.feature_extract=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3),stride=(1,1),padding=(1,1))
        )

        self.sigmoid=nn.Sigmoid()



    def forward(self,real,inpaint):
        #extract features
        real_features=self.feature_extract(real)
        inpaint_features=self.feature_extract(inpaint)

        #add the images as fine features
        real_out=torch.cat((real_features, real), 1)
        inpaint_out=torch.cat((inpaint_features, inpaint), 1)

        #l2 
        combined= torch.sum(torch.pow(real_out - inpaint_out,2),1).unsqueeze(1)
        out=self.sigmoid(combined)
        return out


if __name__=="__main__":

    mymodel=UsiamSimple()
    inp=torch.rand(2,3,512,512)
    print(mymodel(inp,inp).shape)
    summary(mymodel, [(3,288,512), (3,288,512)])