import os
from glob import glob
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import pandas as pd
import torch


class Dataset(object):
    """
    レタスの群落画像のデータセット

    train: Trueで学習用データ，Falseでテスト用データ

    transform: データの前処理を行う関数
    """

    def __init__(self, train=True, transform=None):

        self.train = train
        self.transform = transform
        self.classes = {"normal": 0, "tipburn": 1}

        if os.path.exists("dataset/train.csv") and os.path.exists("dataset/test.csv"):
            if train:
                self.df = pd.read_csv("dataset/train.csv")
            else:
                self.df = pd.read_csv("dataset/test.csv")
        else:
            path_list = glob("dataset/annotations/*.xml")
            df = pd.DataFrame(np.random.permutation(path_list), columns=["path"])
            train_num = int(len(path_list) * 0.8)
            train_df = df[:train_num]
            test_df = df[train_num:].reset_index(drop=True)
            train_df.to_csv("dataset/train.csv")
            test_df.to_csv("dataset/test.csv")
            if train:
                self.df = train_df
            else:
                self.df = test_df

    def __getitem__(self, idx):

        xml_path = self.df.loc[idx, "path"]

        root = ET.parse(xml_path).getroot()
        image_path = root.find("path").text
        filename = root.find("filename").text
        image = Image.open(image_path).convert("RGB")
        boxes = []
        labels = []
        area = []
        iscrowd = []

        for ob in root.iter("object"):

            bndbox = ob.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes[ob.find("name").text])
            area.append((xmax - xmin) * (ymax - ymin))
            iscrowd.append(0)

        if len(boxes) == 0:
            boxes = np.zeros((0, 4))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(idx, dtype=torch.int64),
            "area": torch.tensor(area, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.uint8)
        }

        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target, filename

    def __len__(self):
        return len(self.df)


def collate_fn(batch):
    return tuple(zip(*batch))


def image2numpy(image_file_path):
    """
    画像をnumpy.ndarrayに変換する
        image_file_path: 画像ファイルのパス
    """
    image = Image.open(image_file_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224), Image.ANTIALIAS)
    arrayImg = np.asarray(image).astype(np.float32) / 255.

    return arrayImg


class Dataset_new(object):
    """
    レタスの群落画像のデータセット

    train: Trueで学習用データ，Falseでテスト用データ

    transform: データの前処理を行う関数
    """

    def __init__(self, train=True, transform=None):

        self.train = train
        self.transform = transform
        self.classes = {"normal": 0, "tipburn": 1}

        if os.path.exists("dataset_new/train.csv") and os.path.exists("dataset_new/test.csv"):
            if train:
                self.df = pd.read_csv("dataset_new/train.csv")
            else:
                self.df = pd.read_csv("dataset_new/test.csv")
        else:
            path_list = glob("dataset_new/images/*.png")
            df = pd.DataFrame(np.random.permutation(path_list), columns=["path"])
            train_num = int(len(path_list) * 0.8)
            train_df = df[:train_num]
            test_df = df[train_num:].reset_index(drop=True)
            train_df.to_csv("dataset_new/train.csv")
            test_df.to_csv("dataset_new/test.csv")
            if train:
                self.df = train_df
            else:
                self.df = test_df

    def __getitem__(self, idx):

        xml_path = self.df.loc[idx, "path"]

        root = ET.parse(xml_path).getroot()
        image_path = root.find("path").text
        filename = root.find("filename").text
        image = Image.open(image_path).convert("RGB")
        boxes = []
        labels = []
        area = []
        iscrowd = []

        for ob in root.iter("object"):
            bndbox = ob.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes[ob.find("name").text])
            area.append((xmax - xmin) * (ymax - ymin))
            iscrowd.append(0)

        if len(boxes) == 0:
            boxes = np.zeros((0, 4))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(idx, dtype=torch.int64),
            "area": torch.tensor(area, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.uint8)
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target, filename

    def __len__(self):
        return len(self.df)