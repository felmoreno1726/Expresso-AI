

import torch
import torch.nn as nn
from torchvision.models.video.resnet import Bottleneck, VideoResNet

class SequentialBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 *planes)
        super(SequentialBlock, self).__init__()
        self.conv1 = nn.Sequential(
                conv_builder(inplanes, planes, midplanes, stride),
                nn.BatchNorm1d(planes),
                nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
                conv_builder(inplanes, planes, midplanes, stride),
                nn.BatchNorm1d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(x)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TransferVideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers, 
            stem, num_classes=400,
            inplanes = 512,
            zero_init_residual=False):
        """ Generic resNet video generator but shorter. Expects only 1 conv_makers
        Args:
            block (nn.Module): resNet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): ResNet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        #super(VideoResNet1, self).__init__(block, conv_makers, layers, stem, num_classes, zero_init_residual)
        super(TransferVideoResNet, self).__init__()
        self.inplanes = inplanes
        self.stem = stem()
        
        planes = [256, 512, 1024, 2048]

        model_layers = []
        for i in range(len(layers)):
            model_layers.append(
                #self._make_layer(block, conv_makers[i], inplanes*(i+1), layers[i], stride=1)
                self._make_layer(block, conv_makers[i], planes[i], layers[i], stride=1)
            )
        self.layers = nn.Sequential(*model_layers)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #
        last_planes = planes[len(layers) -1]
        self.fc = nn.Linear(last_planes * block.expansion, num_classes)

        #init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    ###NOTE: this does not work. Do not use zero_init_residual
                    nn.init.constant(m.bn3.weight, 0)


    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for name, m in self.named_modules():
            if name != 'stem':
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        #flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)
        return x
