import os
import unicodedata
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from matplotlib.patches import Rectangle
from torch.utils.data import Dataset, DataLoader
from detectron2.data.detection_utils import read_image


def rect(x1, y1, x2, y2, edgecolor="purple", facecolor=None):
    # Add confidence score
    # plt.gca().add_patch(
    #     Rectangle(
    #         (x1, y1-15),
    #         length_text,
    #         15,
    #         fill=False,
    #         lw=0,
    #         edgecolor="black",
    #         facecolor=edgecolor,
    #     )
    # )
    # plt.text(x1, y1-15, confidence_score)

    plt.gca().add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            lw=2.0,
            edgecolor="black",
            facecolor=facecolor,
        )
    )
    plt.gca().add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            lw=1.5,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
    )


def plot_preds_img(idx, preds, score_threshold=0, conf_boxes=None):
    plt.figure(figsize=(16, 9))
    plt.imshow(preds.images[idx][:, :, ::-1])
    for box in preds.true_boxes[idx]:
        p1, p2 = box["points"]
        rect(p1[0], p1[1], p2[0], p2[1], edgecolor="blue")
    for box, score in zip(preds.pred_boxes[idx], preds.scores[idx]):
        if score < score_threshold:
            continue
        box = box.detach().cpu().numpy()
        rect(box[0], box[1], box[2], box[3], edgecolor="yellow")
    if preds.conf_boxes is not None:
        conf_boxes = preds.conf_boxes
    if conf_boxes is not None:
        for box, score in zip(conf_boxes[idx], preds.scores[idx]):
            if score < score_threshold:
                continue
            box = box.detach().cpu().numpy()
            rect(box[0], box[1], box[2], box[3], edgecolor="purple")
    plt.axis("off")


class TrotterDataset(Dataset):
    def __init__(
        self,
        root="/home/leo/Documents/railconf/dataset",
        transform=None,
        target_transform=None,
    ):
        self.root = root
        self.img_labels = np.load(f"{root}/labels.npy", allow_pickle=True)[()]
        self.img_keys = list(self.img_labels.keys())
        self.img_dir = f"{root}/images/"
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_keys)

    def __getitem__(self, idx):
        key = self.img_keys[idx]
        img_path = os.path.join(self.img_dir, unicodedata.normalize("NFC", key))
        image = read_image(img_path, format="BGR")
        label = self.img_labels[key]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        # labels = list()
        # difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            # labels.append(b[2])
            # difficulties.append(b[3])

        # images = np.stack(images, axis=0) #torch.stack(images, dim=0)

        return (
            images,
            boxes,
        )  # , labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

    def describe(self, n=10):
        for img_name in np.random.permutation(list(self.img_labels.keys()))[:10]:
            print(img_name)
            img = Image.open(
                f"{self.root}/images/{unicodedata.normalize('NFC', img_name)}"
            ).convert("RGB")
            # img = cv2.imread(f"{PATH}/images/{img_name}")[...,::-1] # cv2 has inversed channels
            plt.imshow(img)

            for label in self.img_labels[img_name]:
                p1, p2 = label["points"]
                rect(p1[0], p1[1], p2[0], p2[1])

            plt.axis("off")
            plt.show()

    def get_dataloaders(self, splits, shuffle=False, batch_size=1):
        if isinstance(shuffle, bool):
            shuffle = [shuffle] * len(splits)
        elif not isinstance(shuffle, list):
            raise ValueError(
                f"shuffle is of type {type(shuffle)} but only bool or list of bool is accepted"
            )
        sets = torch.utils.data.random_split(self, splits)
        return [
            DataLoader(
                st,
                batch_size=batch_size,
                shuffle=shuffle[i],
                collate_fn=self.collate_fn,
            )
            for i, st in enumerate(sets)
        ]
