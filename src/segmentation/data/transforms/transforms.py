class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, targets):
        mean = image.mean(dim=(1, 2, 3), keepdim=True) if self.mean is None else self.mean
        std = image.std(dim=(1, 2, 3), keepdim=True) if self.std is None else self.std
        image = (image - mean) / std
        return image, targets

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, targets):
        for t in self.transforms:
            img, targets = t(img, targets)
        return img, targets