from torch import nn
from torchvision.models import resnet50


def load_resnet50():
    """ ResNet50 model """
    model = resnet50()
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=1024, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=1024, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=512, out_features=8, bias=True),
    )
    return model
