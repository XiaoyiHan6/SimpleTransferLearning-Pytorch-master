import torch
import numpy as np


def collate(data):
    imgs = []
    annots = []
    for d in data:
        img, annot = d
        imgs.append(img)
        annots.append(int(annot))
    # all imgs have the same height and width
    # img.shape = 3, height,width
    height = imgs[0].shape[1]
    width = imgs[0].shape[2]
    batch_size = len(imgs)
    padded_imgs = np.zeros((batch_size, 3, height, width),
                           dtype=np.float32)
    for i, img in enumerate(imgs):
        padded_imgs[i, :, :, :] = img
    padded_imgs = torch.from_numpy(padded_imgs)
    targets = np.zeros(batch_size)
    for i, annot in enumerate(annots):
        targets[i] = int(annot)
    annots = torch.from_numpy(targets)
    return padded_imgs, annots
