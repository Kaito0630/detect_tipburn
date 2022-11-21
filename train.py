import time
import warnings
import pandas as pd
import torch

import transforms
from functions import construct_dataloaders, construct_dataloaders_2,construct_model, construct_optimizer, tp_fp_fn, average_precision


def train_one_epoch(model, optimizer, dataloader, device):
    """
    1エポック分の学習を行い，画像1枚あたりのlossを返す

    model: 学習を行うモデル

    optimizer: 学習に用いるoptimizer

    dataloader: 学習用データのdataloader
    """

    model.train()

    total_loss = 0
    data_num = 0

    for images, targets, _ in dataloader:

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss = model(images, targets)
        loss = sum(l for l in loss.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * len(targets)
        data_num += len(targets)

    return total_loss / data_num


def evaluate_model(model, dataloader, device):
    """
    モデルの評価を行い，TPの数, FPの数, FNの数，precision, recall, f1_score, 画像1枚当たりの推論時間を返す

    model: 評価を行うモデル

    dataloader: テスト用データのdataloader
    """

    model.eval()

    inference_time = 0
    data_num = 0
    
    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, targets, _ in dataloader:

            y_true.extend(targets)

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            start_time = time.time()
            y = model(images)
            inference_time += (time.time() - start_time)

            y_pred.extend([{k: v.to('cpu') for k, v in t.items()} for t in y])
            
            data_num += len(targets)

        tp, fp, fn = tp_fp_fn(y_true, y_pred)
        precision = 1.0 if (tp + fp == 0) else tp / (tp + fp)
        recall = 1.0 if (tp + fn == 0) else tp / (tp + fn)
        f1_score = 1.0 if (2 * tp + fp + fn == 0) else 2 * tp / (2 * tp + fp + fn)
        ap = average_precision(y_true, y_pred)

    inference_time /= data_num

    return tp, fp, fn, precision, recall, f1_score, ap, inference_time


def train_and_evalutate(backbone_name, batch_size, num_epochs, num_trials):
    """
    1エポックごとに学習と評価を行い，各指標を記録したcsvファイルと学習後のモデルを保存する
    
    backbone_name: 'resnet50', 'resnet101', 'resnet152', 'wide_resnet50_2', 'wide_resnet101_2, 'resnext50_32x4d', 'resnext101_32x8d'のいずれか

    oprimizer_name: 'sgd', 'adam', 'adamax', 'nadam'のいずれか

    batch_size: バッチサイズ

    num_epochs: 学習を行うエポック数

    num_trials: 試行回数
    """

    print('network: Faster R-CNN')
    print('backbone:', backbone_name)
    print('optimizer: Momentum SGD')
    print('batch_size:', batch_size)
    print('num_epochs:', num_epochs)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('device: GPU')
    else:
        device = torch.device('cpu')
        print('device: CPU')

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    history = {
        'trial': [],
        'epoch': [],
        'loss': [],
        'TP': [],
        'FP': [],
        'FN': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'AP': [],
        'training_time': [],
        'inference_time': []
    }

    for trial in range(1, num_trials + 1):
        print('trial: {}/{}'.format(trial, num_trials))

        # construct dataloader, model and optimizer
        train_dataloader, test_dataloader = construct_dataloaders_2(batch_size, train_transform, test_transform)
        model = construct_model(backbone_name, device)
        optimizer = construct_optimizer(model)

        training_time = 0

        for epoch in range(1, num_epochs + 1):

            # training on the train dataset
            start_time = time.time()
            train_loss = train_one_epoch(model, optimizer, train_dataloader, device)
            training_time += (time.time() - start_time)

            # evaluate on the test dataset
            tp, fp, fn, precision, recall, f1_score, ap, inference_time = evaluate_model(model, test_dataloader, device)

            print(
                'epoch: {0}/{1}'.format(epoch, num_epochs),
                'loss: {0:.4f}'.format(train_loss),
                'precision: {0:.4f}'.format(precision),
                'recall: {0:.4f}'.format(recall),
                'f1_score: {0:.4f}'.format(f1_score),
                'AP: {0:.4f}'.format(ap),
                'training_time: {0:.2f}'.format(training_time),
                'inference_time: {0:.4f}'.format(inference_time)
            )

            history['trial'].append(trial)
            history['epoch'].append(epoch)
            history['loss'].append(train_loss)
            history['TP'].append(tp)
            history['FP'].append(fp)
            history['FN'].append(fn)
            history['precision'].append(precision)
            history['recall'].append(recall)
            history['f1_score'].append(f1_score)
            history['AP'].append(ap)
            history['training_time'].append(training_time)
            history['inference_time'].append(inference_time)

        torch.save(model.to('cpu').state_dict(), 'models/fasterrcnn_' + backbone_name + '_' + str(trial) + '.pth')

    history_df = pd.DataFrame(
        history, 
        columns=['trial', 'epoch', 'loss', 'TP', 'FP', 'FN', 'precision', 'recall', 'f1_score', 'AP', 'training_time', 'inference_time']
    )

    history_df.to_csv('result/fasterrcnn_' + backbone_name + '.csv', index=False)


def main():
    warnings.filterwarnings('ignore')
    num_epochs = 200
    num_trials = 10
    batch_size = 4
    # backbone_names = [
    #     'resnet50', 'resnet101', 'resnet152',
    #     'wide_resnet50_2', 'wide_resnet101_2',
    #     'resnext50_32x4d', 'resnext101_32x8d'
    # ]
    backbone_names = [
        'resnet50'
    ]


    for backbone_name in backbone_names:
        train_and_evalutate(backbone_name, batch_size, num_epochs, num_trials)


if __name__=='__main__':
    main()