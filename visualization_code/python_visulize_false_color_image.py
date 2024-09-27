import scipy.io as sio
import numpy as np
from PIL import Image  
def hsi2rgb(dataset):

    dataset_blu = dataset[:, :, 120]
    dataset_gre = dataset[:, :, 41]
    dataset_red = dataset[:, :, 21]

    nomlize_blu = (dataset_blu - dataset_blu.min()) / (dataset_blu.max() - dataset_blu.min()) * 255
    nomlize_gre = (dataset_gre - dataset_gre.min()) / (dataset_gre.max() - dataset_gre.min()) * 255
    nomlize_red = (dataset_red - dataset_red.min()) / (dataset_red.max() - dataset_red.min()) * 255

    # max_3channel = np.max([dataset_blu.max(),dataset_gre.max(),dataset_red.max()])
    # min_3channel = np.min([dataset_blu.min(),dataset_gre.min(),dataset_red.min()])
    # nomlize_blu = (dataset_blu - min_3channel) / (max_3channel - min_3channel) * 255
    # nomlize_gre = (dataset_gre - min_3channel) / (max_3channel - min_3channel) * 255
    # nomlize_red = (dataset_red - min_3channel) / (max_3channel - min_3channel) * 255

    img_array = np.zeros((nomlize_blu.shape[0], nomlize_blu.shape[1], 3))
    img_array[:, :, 0] = nomlize_blu
    img_array[:, :, 1] = nomlize_gre
    img_array[:, :, 2] = nomlize_red

    return img_array


file_path = 'QUH-Tangdaowan.mat'
data_dic = sio.loadmat(file_path)
dataset = data_dic['Tangdaowan']

array_rgb = hsi2rgb(dataset)
imge_out = Image.fromarray(np.uint8(array_rgb),'RGB')
# imge_out.show()
imge_out = imge_out.save("Tangdaowan-false-color2.png")


