import torch
import torchvision.models as models

class CustomLPIPS:
    def __init__(self):
        self.vgg16 = self.load_frozen_vgg()

        self.feature_layers = self.extract_feature_layers()

        self.vgg16.eval()

    def load_frozen_vgg(self):
        # Load pre-trained VGG16 model
        vgg16 = models.vgg16(pretrained=True)

        # Freeze the weights
        for param in vgg16.parameters():
            param.requires_grad = False

        return vgg16
    
    def extract_feature_layers(self):
        features = list(self.vgg16.features.children())
        interested_layers = []
        for layer in features:
            interested_layers.append(layer)
        return interested_layers
    
    def get_feature_maps(self, x):
        feature_maps = []

        for layer in self.feature_layers:
            x = layer(x)
            feature_maps.append(x)

        return feature_maps
    
    def compute_lpips(self, x1, x2):
        x1_feature_maps = self.get_feature_maps(x1)
        x2_feature_maps = self.get_feature_maps(x2)

        assert len(x1_feature_maps) == len(x2_feature_maps)

        mean_per_layer = []

        for _x1, _x2 in zip(x1_feature_maps, x2_feature_maps):
            mean = torch.mean(torch.square(_x2-_x1)).item()
            mean_per_layer.append(mean)

        return sum(mean_per_layer) / len(mean_per_layer) 



if __name__ == "__main__":
    custom_lpips = CustomLPIPS()

    random_image = torch.rand((1, 3, 128, 128))
    random_image_2 = torch.rand((1, 3, 128, 128))

    print(custom_lpips.compute_lpips(random_image, random_image_2))
