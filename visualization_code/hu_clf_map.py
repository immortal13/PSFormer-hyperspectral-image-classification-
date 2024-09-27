######################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import sklearn.preprocessing

ip_mask=io.loadmat(r'Houston_gt.mat')['gnd_flag']

path='Houston_SVM_grid_pred'
data_mat = io.loadmat(path)
print(data_mat.keys())
data = data_mat['pred_mat'].astype(int)
gt=data.reshape(-1,1)

# # colors for each category in the dataset
hu_colors = np.array([[255, 255, 255],
                               [255, 254, 137], [3,  28,  241], [255, 89,    1], [5,   255, 133],
                               [255,   2, 251], [89,  1,  255], [3,   171, 255], [12,  255,   7],
                               [172, 175,  84], [160, 78, 158], [101, 173, 255], [60,   91, 112],
                               [104, 192,  63], [139, 69,  46], [119, 255, 172], [254, 255,   3],
                               [55,   215, 133], [255, 20, 137], [25,161,95], [221,81,69],
                               ])
ip_colors = sklearn.preprocessing.minmax_scale(hu_colors, feature_range=(0, 1))

# pu_colors = np.array([[255, 255, 255],
#                                 [55,   215, 133],
#                                [255, 20, 137], [3,  28,  241], [255, 89,    1], [5,   255, 133],
#                                [172, 175,  84], [160, 78, 158], [101, 173, 255], [60,   91, 112],
#                                ])
# ip_colors = sklearn.preprocessing.minmax_scale(pu_colors, feature_range=(0, 1))

# sa_colors = np.array([[255, 255, 255],[43,36,242], [43,237,5], [212,49,8], [29,54,115], [32,87,24], 
#     [219,7,138], [245,213,5], [16,232,192], [77,27,15], [176,240,108],
#  [105,98,97], [245,170,91], [167,109,214], [87,180,235], [94,12,66], [4,22,128]])
# ip_colors = sklearn.preprocessing.minmax_scale(sa_colors, feature_range=(0, 1))

# build the RGB Image
clf_map = np.zeros(shape=(data.shape[0], data.shape[1], 3))
cont = 0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if ip_mask[i][j]!=0:
          clf_map[i, j, :] = ip_colors[gt[cont, 0]]
          cont += 1
        else:
            clf_map[i, j, :] = ip_colors[0]
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
