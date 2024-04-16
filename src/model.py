import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18

def remove_layer(model, n):
    return nn.Sequential(*list(model.children())[:-n])

def get_num_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

class RegNet_v2(nn.Module):
    def __init__(self):
        super(RegNet_v2, self).__init__()
        self.RGB_net = nn.Sequential(
            nn.Conv2d(3, 96, 3, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(96, 256, 3, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(256, 384, 3, 1, 0, bias=False),
        )

        self.depth_net = nn.Sequential(
            nn.Conv2d(1, 48, 3, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(48, 128, 3, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(128, 192, 3, 1, 0, bias=False),
        )

        self.matching = nn.Sequential(
            nn.Conv2d(576, 512, 3, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(512, 512, 3, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 8)

    def forward(self, rgb_img, depth_img):
        rgb_features = self.RGB_net(rgb_img)
        depth_features = self.depth_net(depth_img)
        concat_features = torch.cat((rgb_features, depth_features), 1)
        matching_features = self.matching(concat_features).squeeze()
        x = self.fc1(matching_features)
        x = self.fc2(x)

        return x

class RegNet_v1(nn.Module):
    def __init__(self):
        super(RegNet_v1, self).__init__()
        self.RGB_net = self._get_resnet18(pretrained=True)
        self.depth_net = self._get_resnet18(pretrained=False)
        self._adjust_input_channels(self.depth_net, 1)

        for param in self.RGB_net.parameters():
            param.requires_grad = False

        self.matching = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 8)
        )

    def forward(self, rgb_img, depth_img):
        rgb_features = self.RGB_net(rgb_img)
        depth_features = self.depth_net(depth_img)
        concat_features = torch.cat((rgb_features, depth_features), 1)
        matching_features = self.matching(concat_features).squeeze()
        x = self.fc(matching_features)
        return x

    def _get_resnet18(self, pretrained):
        model = resnet18(pretrained=pretrained)
        model = nn.Sequential(*list(model.children())[:-2])
        return model

    def _adjust_input_channels(self, model, in_channels):
        first_conv_layer = model[0]
        if isinstance(first_conv_layer, nn.Conv2d):
            first_conv_layer.in_channels = in_channels
            weight = first_conv_layer.weight.detach().clone()
            new_weight = weight[:, :in_channels, :, :]
            with torch.no_grad():
                first_conv_layer.weight.data = new_weight