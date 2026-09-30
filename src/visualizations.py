import torch 
import matplotlib.pyplot as plt
from src.config import ID2COLOR, CLASS_NAMES, MEAN, STD


def plot_image(image, mask, pred_mask=None, transforms=False, id2color=ID2COLOR):

    image = image.detach().cpu()
    mask = mask.detach().cpu()

    mean = torch.tensor(MEAN).reshape(3,1,1)
    std = torch.tensor(STD).reshape(3,1,1)

    if transforms:
        image = (image * std + mean).clamp(0,1)
    
    colors = torch.tensor(list(id2color.values()), dtype=torch.uint8)

    panels = [
        (image.permute(1,2,0), "Image"),
        (colors[mask.long()], "True Mask"),
    ]

    if pred_mask is not None:
        pred_mask = pred_mask.detach().cpu()
        panels.append((colors[pred_mask.long()], "Pred Mask"))

    fig, axs = plt.subplots(1, len(panels), figsize=(15, 18))
    for ax, (img, title) in zip(axs, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Visulizations.py")