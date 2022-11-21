import os
from PIL import ImageDraw
from PIL import Image
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path

import transforms
from functions import iou, tp_fp_fn, construct_dataloaders,construct_dataloaders_2, construct_model


def draw_and_save(image, target_boxes, tp_boxes, fp_boxes, save_path):
    """
    画像に予測領域を描画して保存する。

    image: 画像

    target_boxes: 正解領域

    tp_boxes: 正しい予測領域

    fp_boxes: 正しくない予測領域

    save_path: 画像の保存先のパス
    """

    # ImageDrawオブジェクト生成
    draw = ImageDraw.Draw(image)
            
    for box in target_boxes:
        """
        ImageDraw.rectangle(xy, fill=None, outline=None, width=1)
        xy	左上と右下の座標をタプルのリスト[(x0, y0), (x1, y1)]
        もしくは x,y座標のリスト[x0, y0, x1, y1]で指定します。
        fill	    領域を塗りつぶす色を指定します。
        outline	    輪郭線の色を指定します。
        width	    線幅を指定します。
        """
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='black', width=3)
        
    for box in tp_boxes:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='blue', width=3)
        
    for box in fp_boxes:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red', width=3)

    image.save(save_path)


def output(model, dataloader, save_dir):
    """
    dataloaderの画像すべてに対して予測領域を計算し，描画して保存する。

    model: 物体検出のモデル

    dataloader: 予測する画像のdataloader

    save_dir: 画像を保存するディレクトリ
    """
    #保存用ディレクトリがなければ作る
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():

        count_box = []

        for images, targets, filenames in dataloader:

            # print("--------------ここまで--------------")
            images = [image.to(device) for image in images]  # GPUへ画像を送る

            y = model(images) #GPUで計算
            # print(y)
            """
            Pythonのenumerate()関数を使うと、forループの中でリストやタプルなどの
            イテラブルオブジェクトの要素と同時にインデックス番号（カウント、順番）を取得できる。
            """
            for i, image in enumerate(images):
                """
                関数 Image.fromarray() は配列オブジェクトを入力として受け取り、その配列オブジェクトから作成した画像オブジェクトを返します。
                """
                image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

                target_boxes = targets[i]['boxes'].data.numpy()

                prediction_boxes = y[i]['boxes'].data.cpu().numpy()
                scores = y[i]['scores'].data.cpu().numpy()
                #labels = y[i]['labels'].data.cpu().numpy()
                # scores = scores[scores >= 0.7]
                prediction_boxes = prediction_boxes[scores >= 0.5].astype(np.int32)


                iou_matrix = iou([targets[i]], [{key: y[i][key].data.cpu() for key in y[i].keys()}])[0]
                
                m = iou_matrix.shape[0]
                n = iou_matrix.shape[1]

                if m == 0:
                    tp = []
                    fp = list(range(n))
                elif n == 0:
                    tp = []
                    fp = []
                else:
                    match = iou_matrix.argmax(dim=0)
                    matched = []
                    tp = []
                    fp = []

                    for j in range(n):
                        if (iou_matrix[match[j], j] >= iou_threshold) and (match[j] not in matched):
                            matched.append(match[j])
                            tp.append(j)
                        else:
                            fp.append(j)

                tp_boxes = prediction_boxes[tp]
                fp_boxes = prediction_boxes[fp]
                save_path = os.path.join(save_dir, filenames[i])

                draw_and_save(image, target_boxes, tp_boxes, fp_boxes, save_path)

                tp, fp, fn = tp_fp_fn([targets[i]], [{key: y[i][key].data.cpu() for key in y[i].keys()}])

                count_box.append({
                    'filename': filenames[i], 
                    'target': len(target_boxes), 
                    'prediction': len(prediction_boxes),
                    'TP': tp,
                    'FP': fp,
                    'FN': fn
                })
            # break

        pd.DataFrame(count_box).to_csv(save_dir + '/count_box.csv')


# backbone_names = [
#     'resnet50', 'resnet101', 'resnet152',
#     'wide_resnet50_2', 'wide_resnet101_2',
#     'resnext50_32x4d', 'resnext101_32x8d'
# ]

backbone_names = [
    'resnet50'
]

for backbone_name in backbone_names:
    model_path = 'models/fasterrcnn_' + backbone_name + '_1.pth'
    iou_threshold = 0.5

    model = construct_model(backbone_name, 'cpu')
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader = construct_dataloaders_2(1, train_transform, test_transform)

    save_dir = 'output/fasterrcnn_' + backbone_name

    # output(model,train_dataloader, save_dir + '/train')
    output(model,test_dataloader, save_dir + '/test')
