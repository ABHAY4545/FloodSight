from pathlib import Path
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode


class FloodNetDataset(Dataset):
    def __init__(self, split, transform=None, root="./FloodNet/FloodNet-Supervised_v1.0"):
        self.transform = transform
        self.image_dir = Path(root) / split / f"{split}-org-img"
        self.mask_dir  = Path(root) / split / f"{split}-label-img"
        self.image_paths = sorted(self.image_dir.iterdir(), key= lambda p: int(p.stem))
        self.mask_paths = [self.mask_dir / f"{f.stem}_lab.png" for f in self.image_paths]
        self.samples = list(zip(self.image_paths, self.mask_paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = read_image(str(image_path), mode=ImageReadMode.RGB)
        mask = read_image(str(mask_path), mode=ImageReadMode.GRAY).squeeze(0)

        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        if self.transform:
            image, mask = self.transform(image, mask)
            
        return image,mask


if __name__ == "__main__":
    print("Dataset.py")
