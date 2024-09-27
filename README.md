# PSFormer-hyperspectral-image-classification
Demo code of "PSFormer: Pyramid Superpixel Transformer for Hyperspectral Image Classification"

## Step 1: prepare dataset
**Salinas and Pavia University**: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

**Houston University (withouot cloud)**: https://github.com/danfenghong/IEEE_TGRS_SpectralFormer

**WHU-Hi-HongHu**: http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm

**QUH-Tangdaowan**: https://github.com/Hang-Fu/QUH-classification-dataset

## Step 2: train and test
```
cd CODE/PSFormer
CUDA_VISIBLE_DEVICES=7 python main.py --tr 30
```
## Citation
If you find this work interesting in your research, please kindly cite:
```
@ARTICLE{10695122,
  author={Zou, Jiaqi and He, Wei and Zhang, Hongyan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={PSFormer: Pyramid Superpixel Transformer for Hyperspectral Image Classification}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2024.3468876}}
```
Thank you very much! (*^▽^*)

This code is constructed based on [MSSG-UNet](https://github.com/qichaoliu/MSSG-UNet), [SuperpixelHierarchy](https://github.com/xingxjtu/SuperpixelHierarchy), and [AM-GCN](https://github.com/zhumeiqiBUPT/AM-GCN), thanks~💕.

If you have any questions, please feel free to contact me (Jiaqi Zou, immortal@whu.edu.cn).
