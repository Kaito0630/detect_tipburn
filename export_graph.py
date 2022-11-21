from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


backbone_names = [
    'resnet50', 'resnet101', 'resnet152',
    'wide_resnet50_2', 'wide_resnet101_2',
    'resnext50_32x4d', 'resnext101_32x8d'
]
legend = {
    'resnet50': 'ResNet50', 'resnet101': 'ResNet101', 'resnet152': 'ResNet152',
    'wide_resnet50_2': 'WideResNet50', 'wide_resnet101_2': 'WideResNet101',
    'resnext50_32x4d': 'ResNeXt50', 'resnext101_32x8d': 'ResNeXt101'
}
y_label = {
    'loss': 'loss', 'precision': 'precision', 'recall': 'recall', 'f1_score': '$F_1$ score', 'AP': 'AP'
}

for metrics in ['loss', 'precision', 'recall', 'f1_score', 'AP']:
    filename = 'graphs/fasterrcnn_' + metrics
    fontsize = 10

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath}'

    plt.figure()

    for backbone_name in backbone_names:
        
        df = pd.read_csv('result/fasterrcnn_' + backbone_name + '.csv')
        plt.plot(np.arange(1, 201), df.groupby('epoch').mean()[metrics], linewidth=1, label=legend[backbone_name])

    plt.xlim([0, 200])
    plt.xticks(np.linspace(0, 200, 5))
    if metrics == 'recall' or metrics == 'f1_score':
        plt.ylim([0, 0.5])
        plt.yticks(np.linspace(0, 0.5, 11))
    elif metrics == 'AP':
        plt.ylim([0, 0.4])
        plt.yticks(np.linspace(0, 0.4, 9))
    else:
        plt.ylim([0, 1])
        plt.yticks(np.linspace(0, 1, 11))
    if metrics == 'loss':
        plt.legend(fontsize=fontsize, loc='upper right')
    else:
        plt.legend(fontsize=fontsize, loc='lower right')
    plt.xlabel('epochs', fontsize=fontsize)
    plt.ylabel(y_label[metrics], fontsize=fontsize)
    #plt.grid()
    plt.savefig(filename + '.svg')
    plt.savefig(filename + '.pdf')
