"""
Model definitions for plant image classification.
Includes: simple CNN, and wrappers for transfer learning (ResNet, etc.)
"""

try:
    import torch
    import torch.nn as nn
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class SimpleCNN(nn.Module):
        """
        Lightweight CNN for plant classification.
        Good for smaller datasets; no pretrained weights.
        """

        def __init__(self, num_classes=3, in_channels=3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x


    def build_model(model_name="simple_cnn", num_classes=3, pretrained=True):
        """
        Build a classification model by name.
        """
        if model_name == "simple_cnn":
            return SimpleCNN(num_classes=num_classes)

        if model_name == "resnet18":
            model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model

        if model_name == "resnet34":
            model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model

        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model

        if model_name == "vgg16":
            model = models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
            model.classifier[6] = nn.Linear(4096, num_classes)
            return model

        raise ValueError(f"Unknown model: {model_name}")
