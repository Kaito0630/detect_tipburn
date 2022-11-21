from pathlib import Path
import cv2
import glob
import os
import numpy as np
from PIL import Image
from natsort import natsorted

def test1():
    img_paths = glob.glob("./dataset_new/valid/images_2/*.jpg")
    for img_path in img_paths:
        img = cv2.imread(img_path)
        rows = 2  # 行数
        cols = 2  # 列数

        chunks = []
        for row_img in np.array_split(img, rows, axis=0):
            for chunk in np.array_split(row_img, cols, axis=1):
                chunks.append(chunk)

        output_dir = Path("./dataset_new/valid/images")
        output_dir.mkdir(exist_ok=True)
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        for i, chunk in enumerate(chunks):
            save_path = output_dir / f"{file_name}{i:02d}.jpg"
            cv2.imwrite(str(save_path), chunk)

def test2():
    # 所定のフォルダ内にある jpg ファイルを連続で読み込んでリスト化する
    files = glob.glob("./images2" + "/*.jpg")

    # 空のリストを準備
    d = []

    # natsortedで自然順（ファイル番号の小さい順）に1個づつ読み込む
    for i in natsorted(files):
        img = Image.open(i)  # img は'JpegImageFile' object
        img = np.asarray(img)  # np.asarrayで img を ndarray に変換
        d.append(img)  # d にappend で img を追加

    # 画像の高さ方向と幅方向を結合
    img_x = np.vstack((np.hstack(d[0:2]),
                       np.hstack(d[2:4]),
                       ))

    # 色をBGR から RGB に変更
    img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
    # cv2.imshow('img_after', img_x)
    cv2.imwrite('./test_image/result.jpg', img_x)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


test1()