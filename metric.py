import numpy as np
import torch

from PIL import Image, ImageDraw
from torchmetrics import JaccardIndex


def to_mask(pred, target):
    pred_boxes = [s['boxes'] for s in pred]  # list with batch_size tensors of shape [num_instances, 4]
    target_boxes = [t['boxes'] for t in target]
    device = pred_boxes[0].device

    for i, sample in enumerate(pred_boxes):
        pred_mask = Image.new('L', (pred[i]['masks'].size(-2), pred[i]['masks'].size(-1)), color=0)
        for j in range(sample.size(0)):
            draw = ImageDraw.Draw(pred_mask)
            draw.rectangle([v.item() for v in sample[j, :]], fill=1)

    for i, sample in enumerate(target_boxes):
        target_mask = Image.new('L', (target[i]['masks'].size(-2), target[i]['masks'].size(-1)), color=0)
        for j in range(sample.size(0)):
            draw = ImageDraw.Draw(target_mask)
            draw.rectangle([v.item() for v in sample[j, :]], fill=1)

    pred_mask, target_mask = torch.tensor(np.array(pred_mask), device=device), torch.tensor(np.array(target_mask), device=device)

    return pred_mask, target_mask


class IntersectionOverUnion(JaccardIndex):
    pass
