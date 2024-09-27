######################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import sklearn.preprocessing

path='WHU_Hi_HongHu_PSFormer_pred_mat'
data_mat = io.loadmat(path)
print(data_mat.keys())
data = data_mat['pred_mat'].astype(int)
gt=data.reshape(-1,1)

pu_colors = np.array([[255, 255, 255],
                               [236,0,0], [255,255,255], [177,41,90], [255,237,0], [255,133,81],
                               [0,198,0], [0,167,0], [0,111,0], [80,226,195], [146,7,194], [218,191,212], 
                               [0,9,191], [0,0,119], [219,108,194], [159,77,35], [0,186,227],
                               [255,165,0], [100,214,0], [128,116,0], [0,96,125], [208,181,203], [247,144,0], 
                               ])
ip_colors = sklearn.preprocessing.minmax_scale(pu_colors, feature_range=(0, 1))

# build the RGB Image
clf_map = np.zeros(shape=(data.shape[0], data.shape[1], 3))
cont = 0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        # if ip_mask[i][j]!=0:
        #   clf_map[i, j, :] = ip_colors[gt[cont, 0]]
        #   cont += 1
        # else:
        #     clf_map[i, j, :] = ip_colors[0]
        #     cont += 1

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
