from torch import nn
from torchvision.models import mobilenet_v3_large


def load_mobilenetv3large():
    """ Mobilenet_v3_large model """
    model = mobilenet_v3_large()
    model.classifier = nn.Sequential(
        nn.Linear(in_features=960, out_features=1280, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=1280, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=512, out_features=8, bias=True)
    )
    return model
