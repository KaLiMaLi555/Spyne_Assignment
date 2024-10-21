import torch
from torch import nn
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms

from .mobilenetv3large import load_mobilenetv3large
from .resnet50 import load_resnet50
from .vgg16 import load_vgg16


def load_model(model_name, device):
    """ Load the model from the checkpoints folder """
    model_path = f"checkpoints/{model_name}.pth"
    if model_name == "vgg16":
        model = load_vgg16()
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    elif model_name == "resnet50":
        model = load_resnet50()
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    elif model_name == "mobilenet_v3_large":
        model = load_mobilenetv3large()
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    else:
        raise ValueError("Invalid model name")
    return model


def evaluate_model(model_name, image_path):
    """ Evaluate the model on the given image """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, device)

    # Set the model to evaluation mode
    model.eval()

    # Read the image
    test_tensor = read_image(image_path)

    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_tensor = transform(test_tensor)

    # Evaluate the model
    output = model(test_tensor.unsqueeze(0).to(device))

    # Get the confidence and class
    confidences = nn.functional.softmax(output, dim=1)
    _, preds = torch.max(output, 1)
    classes = ["0", "40", "90", "130", "180", "230", "270", "320"]

    pred_class = classes[int(preds[0])]
    confidence = float(confidences[0, int(preds[0])].data)

    return pred_class, confidence
