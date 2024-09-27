######################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import sklearn.preprocessing

ip_mask=io.loadmat(r'QUH-Tangdaowan_GT')['TangdaowanGT']

path='QUH-Tangdaowan_GT'
data_mat = io.loadmat(path)
data = data_mat['TangdaowanGT'].astype(int)
print(np.unique(data))
gt=data.reshape(-1,1)

# # colors for each category in the dataset
hu_colors = np.array([[255, 255, 255],
                               [140,67,46], [153,153,153], [255,100,0], [0,255,123],
                               [164,75,155], [101,174,255], [118,254,172], [60,91,112],[255,255,0], 
                               [255,255,125], [255,0,255], [100,0,255], [0,172,254], 
                               [0,255,0], [171,175,80], [101,193,60], [139,0,0], [0,0,255],
                               ])
ip_colors = sklearn.preprocessing.minmax_scale(hu_colors, feature_range=(0, 1))

# build the RGB Image
clf_map = np.zeros(shape=(data.shape[0], data.shape[1], 3))
cont = 0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        # if ip_mask[i][j]!=0:
        #   clf_map[i, j, :] = ip_colors[gt[cont, 0]]
        #   cont += 1
        # else:
        clf_map[i, j, :] = ip_colors[gt[cont, 0]]
        cont += 1

def Draw_Classification_Map(label, name: str, dpi: int = 400):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(label.shape[1] * 2.0 / dpi,
                        label.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(label)
    fig.savefig(name + '.png', dpi=dpi)

Draw_Classification_Map(clf_map,path)
