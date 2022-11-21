import pandas as pd


backbone_names = [
    'resnet50', 'resnet101', 'resnet152',
    'wide_resnet50_2', 'wide_resnet101_2',
    'resnext50_32x4d', 'resnext101_32x8d'
]

df_list = []

for backbone_name in backbone_names:
    csv_path = 'result/fasterrcnn_' + backbone_name + '.csv'
    df = pd.read_csv(csv_path)
    df = df[df['epoch'] == 200].mean()
    df['backbone'] = backbone_name
    df['precision'] *= 100
    df['recall'] *= 100
    df['f1_score'] *= 100
    df['AP'] *= 100
    df['training_time'] /= 60
    df['inference_time'] *= 1000
    df = df[['backbone', 'precision', 'recall', 'f1_score', 'AP', 'training_time', 'inference_time']]
    df_list.append(df)

df = pd.DataFrame(df_list)
print(df)