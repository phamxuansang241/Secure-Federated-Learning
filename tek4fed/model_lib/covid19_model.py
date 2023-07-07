from torchvision.models import resnet18
import torch


def pretrain_resnet18():
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=3)

    return model
