
import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.LeakyReLU(0.01)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.LeakyReLU(0.01)
        self.conv3 = nn.Conv1d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels * 4)
        self.relu3 = nn.LeakyReLU(0.01)

        if in_channels != out_channels * 4:
            self.identity = nn.Conv1d(in_channels, out_channels * 4, kernel_size=1, stride=stride, padding=0)
            self.bn_id = nn.BatchNorm1d(out_channels * 4)
        else:
            self.identity = None

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        if self.identity is not None:
            identity = self.identity(identity)
            identity = self.bn_id(identity)

        x += identity
        x = self.relu3(x)

        return x
    
class ResNet101Regressor(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResNet101Regressor, self).__init__()
        self.conv1 = nn.Conv1d(input_size[1], 64, kernel_size=7, stride=2, padding=3)
        
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.block1 = BottleneckBlock(64, 64)
        self.block2 = BottleneckBlock(256, 64)
        self.block3 = BottleneckBlock(256, 64)

        self.block4 = BottleneckBlock(256, 128, stride=2)
        self.block5 = BottleneckBlock(512, 128)
        self.block6 = BottleneckBlock(512, 128)
        self.block7 = BottleneckBlock(512, 128)

        self.block8 = BottleneckBlock(512, 256, stride=2)
        self.block9 = BottleneckBlock(1024, 256)
        self.block10 = BottleneckBlock(1024, 256)
        self.block11 = BottleneckBlock(1024, 256)
        self.block12 = BottleneckBlock(1024, 256)
        self.block13 = BottleneckBlock(1024, 256)

        self.block14 = BottleneckBlock(1024, 512, stride=2)
        self.block15 = BottleneckBlock(2048, 512)
        self.block16 = BottleneckBlock(2048, 512)

        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(2048, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)

        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)

        x = self.global_avg_pooling(x)
        x = x.view(-1, 2048)
        x = self.fc(x)

        return x
