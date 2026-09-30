import torch
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2, InterpolationMode

train_transforms = v2.Compose([
    v2.Resize((384,512)),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomAffine(
        degrees=(-10, 10),
        translate=(0.1,0.1),
        scale=(0.9,1.1),
        interpolation=InterpolationMode.BILINEAR
    ),
    v2.ToDtype({
        tv_tensors.Image: torch.float32,
        tv_tensors.Mask : torch.int64
    }, scale=True),
    v2.Normalize(mean=(0.4102, 0.4471, 0.3415),
    std=(0.2085, 0.1936, 0.2092)),
    v2.ToPureTensor()
])

val_transforms = v2.Compose([
    v2.Resize((384, 512)),
    v2.ToDtype({
        tv_tensors.Image: torch.float32,
        tv_tensors.Mask: torch.int64
    }, scale=True),
    v2.Normalize(mean=(0.4102, 0.4471, 0.3415),
    std=(0.2085, 0.1936, 0.2092)),
    v2.ToPureTensor()
])

test_transforms = val_transforms

if __name__ == "__main__":
    print("Transforms.py")
