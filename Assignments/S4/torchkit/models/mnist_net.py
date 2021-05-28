import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()

        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 28 OUtput - 26 RF - 3
                                   
                                  nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(16), #input - 26 OUtput - 24 RF - 5
                                   
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(32) #input - 24 OUtput - 22 RF - 7
        )


       
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
                                  nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1,1)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 11 OUtput - 11 RF - 12
                                   
                                  nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(16), #input - 11 OUtput - 9 RF - 16
                                   
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(32) #input - 9 OUtput - 7 RF - 20
        )
        
        self.conv3 = nn.Conv2d(32, 64, 1) # input - 7 - 20
        
        self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,10)

        self.gap = nn.AvgPool2d(7)

    def forward(self, x):
        conv1_op = self.conv1(x)
        conv1_op = F.dropout(conv1_op, p=0.030)

        pool1_op = self.pool1(conv1_op)

        conv2_op = self.conv2(pool1_op)
        conv2_op = F.dropout(conv2_op, p=0.030)

        gap_op = self.gap(conv2_op)
        conv3_op = self.conv3(gap_op)
        
        conv3_op = conv3_op.view(conv3_op.shape[0],-1)
        
        fc1_op = self.fc1(conv3_op)
        fc2_op = self.fc2(fc1_op)

        final_op = fc2_op.view(-1, 10)
        return F.log_softmax(final_op, dim=-1)
      
