import torch

def compute_tp_fp_fn(pred_mask, true_mask, num_classes = 10):

    assert pred_mask.ndim == 4, "Prediction should have 4 dimensions"
    assert pred_mask.shape[1] == num_classes, "Number of classes does not match the number of channels in pred_mask"
    assert true_mask.ndim == 3, "Mask should have 3 dimensions"

    pred_mask = torch.argmax(pred_mask, dim=1).flatten()
    true_mask = true_mask.to(device=pred_mask.device, dtype=torch.long).flatten()

    confusion = torch.bincount( true_mask * num_classes + pred_mask, minlength=num_classes ** 2).reshape(num_classes, num_classes)

    tps = confusion.diag()
    fps = confusion.sum(dim=0) - tps
    fns = confusion.sum(dim=1) - tps

    return tps, fps, fns

def dice_score(tps,fps,fns):

    valid = tps+ fps + fns> 0




    return (2*tps.sum()) / (2*tps.sum()+fps.sum()+fns.sum())
     