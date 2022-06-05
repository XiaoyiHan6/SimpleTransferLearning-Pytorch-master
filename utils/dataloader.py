import os
from PIL import Image
from .path import PETS_path
from torch.utils.data import Dataset


class pets(Dataset):
    def __init__(self, root_dir, label_dir='annots', transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_path = os.path.join(self.root_dir, 'imgs')
        self.img_path = os.listdir(self.img_path)
        self.label_dir = os.path.join(self.root_dir, self.label_dir, 'label.txt')

    def __getitem__(self, idx):
        annot = self.load_annot(idx)
        img = self.load_image(idx)
        if self.transform:
            img = self.transform(img)

        return img, annot

    def load_annot(self, idx):
        with open(self.label_dir, "r") as f:
            for line in f.readlines():
                img_name = self.img_path[idx]
                line = line.strip()
                if img_name in line:
                    # image_name : 101.jpg , category_id : 1 , category : dog;
                    # line[37] = index "1"
                    annot = line[37]
        return annot

    def load_image(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, 'imgs', img_name)
        img = Image.open(img_item_path)
        return img

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    root_dir = PETS_path
    dataset = pets(root_dir=root_dir)
    img, annot = dataset[3]
    print("img.shape: {}, annot: {}".format(img, annot))
