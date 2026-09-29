from torch import nn


def build_model(name, classes):
    if name == "resnet18":
        from torchvision.models import resnet18
        model = resnet18(weights=None, num_classes=classes)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    if name == "tiny":
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, classes),
        )
    raise ValueError(f"unknown model: {name}")
