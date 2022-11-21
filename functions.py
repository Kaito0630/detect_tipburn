import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from dataset_2 import Dataset_2, collate_fn
from dataset import Dataset, collate_fn


def construct_dataloaders(batch_size, train_transform=None, test_transform=None):
    """
    dataloaderを生成する

    batch_size: バッチサイズ

    train_transform: 学習時の前処理を行う関数

    test_transform: テスト時の前処理を行う関数
    """

    train_dataset = Dataset(train=True, transform=train_transform)
    test_dataset = Dataset(train=False, transform=test_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, collate_fn=collate_fn
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, test_dataloader

def construct_dataloaders_2(batch_size, train_transform=None, test_transform=None):
    """
    dataloaderを生成する

    batch_size: バッチサイズ

    train_transform: 学習時の前処理を行う関数

    test_transform: テスト時の前処理を行う関数
    """

    train_dataset = Dataset_2(train=True, transform=train_transform)
    test_dataset = Dataset_2(train=False, transform=test_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, collate_fn=collate_fn
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, test_dataloader



def construct_model(backbone_name, device):
    """
    モデルを生成する

    backbone_name: 'resnet50', 'resnet101', 'resnet152', 'wide_resnet50_2', 'wide_resnet101_2, 'resnext50_32x4d', 'resnext101_32x8d'のいずれか
    
    device: CPUを使用する場合は'cpu'，GPUを使用する場合は'cuda'
    """
    num_classes = 2  # normal or tipburn
    backbone = resnet_fpn_backbone(backbone_name, pretrained=True)
    model = FasterRCNN(backbone, num_classes)
    model.to(device)
    return model


def construct_optimizer(model):
    """
    optimizerを生成する

    model: 学習を行うモデル
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    return optimizer


def iou(y_true, y_pred, score_th=0.1):
    """
    画像ごとに各正解領域と各予測領域のIOUの行列を計算し，IOUの行列のリストを返す

    y_true: 正解データ

    y_pred: 予測データ

    score_th: 信頼度スコアの閾値
    """

    iou_list = []

    for i in range(len(y_true)):

        true_boxes = y_true[i]['boxes'] #i番目の画像の正解領域
        pred_boxes = y_pred[i]['boxes'][y_pred[i]['scores'] >= score_th] #i番目の画像の予測領域

        m = true_boxes.shape[0] #正解領域の個数
        n = pred_boxes.shape[0] #予測領域の個数

        if (m == 0) or (n == 0):
            iou_list.append(torch.zeros(size=(m, n)))

        else:
            intersection_xmin = torch.max(true_boxes[:, 0].expand(n, m).T, pred_boxes[:, 0].expand(m, n))
            intersection_ymin = torch.max(true_boxes[:, 1].expand(n, m).T, pred_boxes[:, 1].expand(m, n))
            intersection_xmax = torch.min(true_boxes[:, 2].expand(n, m).T, pred_boxes[:, 2].expand(m, n))
            intersection_ymax = torch.min(true_boxes[:, 3].expand(n, m).T, pred_boxes[:, 3].expand(m, n))

            intersection_width = torch.max(intersection_xmax - intersection_xmin, torch.zeros(m, n))
            intersection_height = torch.max(intersection_ymax - intersection_ymin, torch.zeros(m, n))
            intersection = intersection_width * intersection_height

            true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
            pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
            union = true_area.expand(n, m).T + pred_area.expand(m, n) - intersection

            iou_list.append(intersection / union)

    return iou_list


def tp_fp_fn(y_true, y_pred, iou_th=0.5):
    """
    正解データと予測データからTP, FP, FNの数を計算する

    y_true: 正解データ

    y_pred: 予測データ

    iou_th: 正解とみなすIOUの閾値
    """

    tp = 0
    fp = 0
    fn = 0

    iou_list = iou(y_true, y_pred)

    for iou_matrix in iou_list:

        m = iou_matrix.shape[0] #正解領域の個数 (TP + FN)
        n = iou_matrix.shape[1] #予測領域の個数 (TP + FP)

        true_posivite = 0

        if m == 0:
            fp += n

        elif n == 0:
            fn += m

        else:
            max_idx = iou_matrix.argmax(dim=0)
            matched = []

            for j in range(n):
                if (iou_matrix[max_idx[j], j] >= iou_th) and (max_idx[j] not in matched):
                    matched.append(max_idx[j])
                    true_posivite += 1

            tp += true_posivite
            fp += (n - true_posivite)
            fn += (m - true_posivite)

    return tp, fp, fn


def average_precision(y_true, y_pred, iou_th=0.5, num_points=None):
    """
    正解データと予測データからAverage Precision (AP)を計算する。

    y_true: 正解データ

    y_pred: 予測データ

    iou_th: 正解とみなすIOUの閾値

    num_points: APの計算に用いる点の数。Noneの場合，すべての点を用いる
    """

    iou_list = iou(y_true, y_pred, score_th=0)
    y_correct = []
    y_scores = []
    n_gt = 0

    for i, iou_matrix in enumerate(iou_list):
        
        m = iou_matrix.shape[0]
        n = iou_matrix.shape[1]

        n_gt += m

        if m == 0:

            for j in range(n):
                y_correct.append(0)
                y_scores.append(y_pred[i]['scores'][j])

        elif n > 0:

            max_idx = iou_matrix.argmax(dim=0)
            matched = []

            for j in range(n):
                if (iou_matrix[max_idx[j], j] >= iou_th) and (max_idx[j] not in matched):
                    matched.append(max_idx[j])
                    y_correct.append(1)
                else:
                    y_correct.append(0)
                y_scores.append(y_pred[i]['scores'][j])

    y_correct = torch.Tensor(y_correct)
    y_scores = torch.Tensor(y_scores)

    if len(y_scores) == 0:
        return 0

    sort_idx = torch.argsort(y_scores, descending=True)
    y_correct_sorted = y_correct[sort_idx]
    cumsum = torch.cumsum(y_correct_sorted, dim=0)
    recall = cumsum / n_gt
    precision = cumsum / torch.arange(1, 1 + y_correct.shape[0])

    if num_points:
        p = torch.zeros(num_points)
        for i in range(num_points - 1, -1, -1):
            idx = (recall >= i / (num_points - 1))
            if torch.any(idx):
                p[i] = torch.max(precision[idx], dim=0)[0]
            else:
                p[i] = 0
        return float(torch.mean(p))
    else:
        ap = 0
        p_max = 0
        r = 1
        for i in range(len(recall) - 1, -1, -1):
            ap += p_max * (r - recall[i])
            p_max = max(precision[i], p_max)
            r = recall[i]
        ap += r
        return float(ap)
