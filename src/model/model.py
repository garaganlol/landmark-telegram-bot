import torch
import torch.nn as nn
from torchvision import models

class BaselineCNN(nn.Module): # Custom VGG
    '''
    VGG подобная сеть: 
    3 слоя:conv2d, batchnorm, relu, conv2d, batchnorm, relu, maxpool
    Перед fc слоем AdaptiveAvgPool и Dropout
    fc слой с 512 нейронами

    Аргументы:
        num_classes(int): кол-во классов для классификации
    '''
    def __init__(self, num_classes):
      super().__init__()

      # 224 * 224 * 3
      self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

      # 112 * 112 * 64
      self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

      # 56 * 56 * 128
      self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(0.25))
      
      # 1 * 1 * 256
      self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1 * 1 * 256, 512),
            nn.ReLU())
      self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out
    
def mobilenet_model(num_classes, pretrained=True, freeze_features=True, device="cpu"):
    """
    
    Аргументы:
        num_classes (int): количество классов в вашем датасете
        pretrained (bool): использовать предобученные веса ImageNet
        freeze_features (bool): заморозить feature extractor для ускорения обучения
        device (str): 'cpu' или 'cuda'
        
    Возвращает:
        model (nn.Module): готовая к обучению модель на указанном устройстве
    """
    if pretrained:
        mobilenet_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    else:
        mobilenet_model = models.mobilenet_v2(weights=None)

    if freeze_features:
        for param in mobilenet_model.features.parameters():
            param.requires_grad = False

    in_features = mobilenet_model.classifier[1].in_features

    mobilenet_model.classifier[1] = nn.Linear(in_features, num_classes)

    mobilenet_model = mobilenet_model.to(device)

    return mobilenet_model