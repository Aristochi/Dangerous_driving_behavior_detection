from torch import nn
import torch.nn.functional as F

class mini_xception(nn.Module):
    def __init__(self):
        super(mini_xception,self).__init__()
        self.num_channels=1
        self.image_size=48
        self.num_labels=7
        self.conv2d_1 =nn.Conv2d(in_channels=46,out_channels=8,kernel_size=3,stride=1)
        self.batch_normalization_1=nn.BatchNorm1d(46)
        self.conv2d_2=nn.Conv2d(46,8,3,1)
        self.batch_normalization_2=nn.BatchNorm1d(46)

        #module 1




    def forward(self, x):
        x=F.relu(self.batch_normalization_1)
        x=F.relu(self.batch_normalization_2)
        return x

if __name__ == '__main__':
    print(mini_xception())