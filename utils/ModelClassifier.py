# model_classifier.py

import torch
import yaml
from torchvision import transforms
from resnet import Bottleneck, ResNet, image_classification
from PIL import Image

class ModelClassifier:
    def __init__(self, config_path="./config/config.yaml"):
        self.config = self._load_config(config_path)
        self.model = self._load_model()
        self.transform = self._build_transform()

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _load_model(self):
        cfg = self.config["resnet"]
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=cfg["num_classes"])
        checkpoint_path = cfg["ckpt"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    def _build_transform(self):
        return transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
        ])

    def __call__(self, image: Image.Image):
        cfg = self.config["resnet"]
        return image_classification(
            model=self.model,
            image=image,
            target_class_idx=cfg["target_label"],
            transform=self.transform,
        )

if __name__ == "__main__":
    classifier = ModelClassifier(config_path="./config/config.yaml")
    matched_paths = classifier()
