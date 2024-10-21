from torch import nn
from torchvision.models import vgg16


def load_vgg16():
    """ VGG16 Model """
    model = vgg16()
    model.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=1024, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=1024, out_features=8, bias=True)
    )
    return model
