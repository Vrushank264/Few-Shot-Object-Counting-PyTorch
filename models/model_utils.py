import torch
import torch.nn as nn
import torch.nn.functional as fun
import torchvision as tv


class ResNet(nn.Module):

    def __init__(self, depth, out_stride, out_layers):

        super().__init__()
        self.out_stride = out_stride
        self.out_layers = out_layers

        assert depth in ['18', '50']

        if depth == '18':
            base_dim = 64
            self.backbone = tv.models.resnet18(False)
        
        elif depth == '50':
            base_dim = 256
            self.backbone = tv.models.resnet50(False)

        children = list(self.backbone.children())
        for idx, c in enumerate(children):
            print(idx, c)
            print('*'*100)
        self.layer0 = nn.Sequential(*children[:4])
        self.layer1 = children[4]
        self.layer2 = children[5]
        self.layer3 = children[6]
        self.layer4 = children[7]

        planes = [base_dim * x for x in range(1, 5)]
        print(planes)

        self.out_dim = sum([planes[i-1] for i in self.out_layers])

    def forward(self, x):

        x = self.layer0(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        features = [feat1, feat2, feat3, feat4]
        out_strides = [4, 8, 16, 32]

        feature_ls = []
        for layer in self.out_layers:
            scale_factor = out_strides[layer - 1] / self.out_stride
            feature = features[layer - 1]
            features = fun.interpolate(feature, scale_factor=scale_factor, mode = 'bilinear')
            feature_ls.append(feature)

        feature = torch.cat(feature_ls, dim = 1)
        return feature


model = ResNet('18', 4, [1,2,3])

