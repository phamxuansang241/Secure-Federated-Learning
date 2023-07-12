from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
import torch
from torchvision.models import vgg19, VGG19_Weights



def pretrain_vgg19():
    weights = VGG19_Weights.DEFAULT
    model = vgg19(weights=weights, progress=False)

    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 64),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(64, 3)
    )

    return model


def pretrain_resnet18():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights, progress=False)
    
    for param in model.parameters():
        param.requires_grad = False

    
    # model.fc = torch.nn.Linear(in_features=512, out_features=3)
    
    for param in model.layer4.parameters():
        param.requires_grad = True
            
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)

    return model

class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y_pred = self.resnet18(x)
            return y_pred.argmax(dim=1)