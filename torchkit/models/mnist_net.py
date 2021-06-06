import torch
import torch.nn as nn
import torch.nn.functional as F



class Skeleton(nn.Module):
    def __init__(self):
        super(Skeleton, self).__init__()
        
        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3)),   
                                #input - 28 OUtput - 26 RF - 3
                                   
                                  nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3)),   
                                    #input - 26 OUtput - 24 RF - 5
                                   
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3)),   
                                #input - 24 OUtput - 22 RF - 7
        )

       
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
                                  nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1,1)),   
                                    #input - 11 OUtput - 11 RF - 12
                                   
                                  nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3)),   
                                    #input - 11 OUtput - 9 RF - 16
                                   
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3)),   
                                 #input - 9 OUtput - 7 RF - 20
        )
        
        self.conv3 = nn.Conv2d(32, 64, 1) # input - 7 - 20
        
        self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,10)

        self.gap = nn.AvgPool2d(7)

    def forward(self, x):
        conv1_op = self.conv1(x)
        pool1_op = self.pool1(conv1_op)
        conv2_op = self.conv2(pool1_op)
        gap_op = self.gap(conv2_op)
        conv3_op = self.conv3(gap_op)
    
        conv3_op = conv3_op.view(conv3_op.shape[0],-1)
        
        fc1_op = self.fc1(conv3_op)
        fc2_op = self.fc2(fc1_op)

        final_op = fc2_op.view(-1, 10)
        return F.log_softmax(final_op, dim=-1)
      
        
    
class BasicMNISTNet(nn.Module):
    def __init__(self):
        super(BasicMNISTNet, self).__init__()

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
      



class AvgMNISTNet(nn.Module):
    def __init__(self):
        super(AvgMNISTNet, self).__init__()

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
      



# Reached 99 in 7-8th Epoch
class DilatedNet(nn.Module):
    def __init__(self, dropout_value = 0.069):
        super(DilatedNet, self).__init__()

        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), padding=1),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 28 OUtput - 28 RF - 3
                                  nn.Dropout(dropout_value),
                                   
                                  nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(12), #input - 28 OUtput - 26 RF - 5
                                  nn.Dropout(dropout_value),
                                   
                                  nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3,3), dilation=2),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(16), #input - 26 OUtput - 22 RF - 9
                                  nn.Dropout(dropout_value),
                                   
                             
        )


       
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
                                  nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1,1)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 11 OUtput - 11 RF - 10
                                  nn.Dropout(dropout_value),
                                   
                                  nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3,3), dilation=2),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(12), #input - 11 OUtput - 7 RF - 18
                                  nn.Dropout(dropout_value),
                                   
                                  nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3,3)),
                                    #input - 7 OUtput - 5 RF - 22
                                   
                                  # nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3,3)),
                                  # nn.ReLU(), 
                                  # nn.BatchNorm2d(24) #input - 7 OUtput - 5 RF - 22
        )
        
        self.conv3 = nn.Conv2d(16, 48, 1) # input - 1  output - 1 RF - 22s
        
        self.fc1 = nn.Linear(48,10)
        # self.fc2 = nn.Linear(32,10)

        self.gap = nn.AvgPool2d(5)

    def forward(self, x):
        conv1_op = self.conv1(x)
        # conv1_op = F.dropout(conv1_op, p=0.050)

        pool1_op = self.pool1(conv1_op)

        conv2_op = self.conv2(pool1_op)
        # conv2_op = F.dropout(conv2_op, p=0.050)

        gap_op = self.gap(conv2_op)
        conv3_op = self.conv3(gap_op)
        
        conv3_op = conv3_op.view(conv3_op.shape[0],-1)
        
        fc1_op = self.fc1(conv3_op)
        # fc2_op = self.fc2(fc1_op)

        final_op = fc1_op.view(-1, 10)
        return F.log_softmax(final_op, dim=-1)
      




class DropNet(nn.Module):
    def __init__(self, dropout_value=0.069):
        super(DropNet, self).__init__()

        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), padding=1),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 28 OUtput - 28 RF - 3
                                   nn.Dropout(dropout_value),
                                   
                                  nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(12), #input - 28 OUtput - 26 RF - 5
                                   nn.Dropout(dropout_value),
                                   
                                  nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(16), #input - 26 OUtput - 24 RF - 7
                                   nn.Dropout(dropout_value)
        )


       
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
                                  nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1,1)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 12 OUtput - 12 RF - 8
                                  nn.Dropout(dropout_value),
                                   
                                  nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(12), #input - 12 OUtput - 10 RF - 12
                                   nn.Dropout(dropout_value),
                                   
                                  nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(16),  #input - 10 OUtput - 8 RF - 16
                                   
                                   
                                  nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3,3)),
                                  # nn.ReLU(), 
                                  # nn.BatchNorm2d(32) #input - 8 OUtput - 6 RF - 20
        )
        
        self.conv3 = nn.Conv2d(20, 32, 1) # input - 8 - 20
        
        self.fc1 = nn.Linear(32,10)
        # self.fc2 = nn.Linear(32,10)

        self.gap = nn.AvgPool2d(6)

    def forward(self, x):
        conv1_op = self.conv1(x)
        # conv1_op = F.dropout(conv1_op, p=0.050)

        pool1_op = self.pool1(conv1_op)

        conv2_op = self.conv2(pool1_op)
        # conv2_op = F.dropout(conv2_op, p=0.050)

        gap_op = self.gap(conv2_op)
        conv3_op = self.conv3(gap_op)
        
        conv3_op = conv3_op.view(conv3_op.shape[0],-1)
        
        fc1_op = self.fc1(conv3_op)
        # fc2_op = self.fc2(fc1_op)

        final_op = fc1_op.view(-1, 10)
        return F.log_softmax(final_op, dim=-1)
      



# increasing model capacity
class NonDilatedNet(nn.Module):
    def __init__(self, dropout_value=0.069):
        super(NonDilatedNet, self).__init__()

        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), padding=1),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 28 OUtput - 28 RF - 3
                                   nn.Dropout(dropout_value),
                                   
                                  nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3,3), padding=1),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(10), #input - 28 OUtput - 28 RF - 5
                                   nn.Dropout(dropout_value),
                                   
                              
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
                                  nn.Conv2d(in_channels=10, out_channels=8, kernel_size=(1,1)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 14 OUtput - 14 RF - 6
                                  nn.Dropout(dropout_value),
                                   
                                  nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3,3), padding=1),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(10), #input - 14 OUtput - 14 RF - 10
                                  nn.Dropout(dropout_value),
                                   
                                  nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(12),  #input - 14 OUtput - 12 RF - 14
                                   nn.Dropout(dropout_value),
                                   
                                  nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(14), #input - 12 OUtput - 10 RF - 18
                                  nn.Dropout(dropout_value),

                                  nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3,3)),
                                   #input - 10 OUtput - 8 RF - 22
        
                                  
        )
        
        self.conv3 = nn.Conv2d(16, 32, 1) # input - 8 - 20
        
        self.fc1 = nn.Linear(32,10)
        # self.fc2 = nn.Linear(32,10)

        self.gap = nn.AvgPool2d(8)

    def forward(self, x):
        conv1_op = self.conv1(x)
       
        pool1_op = self.pool1(conv1_op)

        conv2_op = self.conv2(pool1_op)
       

        gap_op = self.gap(conv2_op)
        conv3_op = self.conv3(gap_op)
        
        conv3_op = conv3_op.view(conv3_op.shape[0],-1)
        
        fc1_op = self.fc1(conv3_op)
        # fc2_op = self.fc2(fc1_op)

        final_op = fc1_op.view(-1, 10)
        return F.log_softmax(final_op, dim=-1)
      
