import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.ch_norm = self.config.channel_norm

    def convblock(self, in_ch=1, mid_ch=8, out_ch=16, kernel_=(3,3), padding_=[0,0], bias=False):#, ch_norm = self.ch_norm):
        if self.ch_norm == 'BatchNorm2d':
            _block = nn.Sequential(
                                            nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=kernel_, padding=padding_[0], bias=False),
                                            nn.ReLU(), 
                                            nn.BatchNorm2d(mid_ch), 
                                            nn.Dropout(self.config.dropout_value),
                                            
                                            nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=kernel_, padding=padding_[1],bias=False),
                                            nn.ReLU(), 
                                            nn.BatchNorm2d(out_ch),
                                            nn.Dropout(self.config.dropout_value)
            )
        if self.ch_norm == 'GroupNorm':
            _block = nn.Sequential(
                                        nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=kernel_, padding=padding_[0], bias=False),
                                        nn.ReLU(), 
                                        nn.GroupNorm(2, mid_ch), 
                                        nn.Dropout(self.config.dropout_value),
                                        
                                        nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=kernel_, padding=padding_[1],bias=False),
                                        nn.ReLU(), 
                                        nn.GroupNorm(2, out_ch),
                                        nn.Dropout(self.config.dropout_value)
            )
        if self.ch_norm == 'LayerNorm':
            _block = nn.Sequential(
                                        nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=kernel_, padding=padding_[0], bias=False),
                                        nn.ReLU(), 
                                        nn.GroupNorm(1, mid_ch),
                                        nn.Dropout(self.config.dropout_value),
                                        
                                        nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=kernel_, padding=padding_[1],bias=False),
                                        nn.ReLU(), 
                                        nn.GroupNorm(1, out_ch),
                                        nn.Dropout(self.config.dropout_value)
            )
        return _block
                                    
  


class CifarNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = CifarNet()