import numpy as np
import scipy.io as sio

class SuperHierarchy(object):
    def __init__(self, datasetname, first_sp_num=2048):
        if datasetname == 'salinas_':
            segs = sio.loadmat('Superpixels/' + 'salinas_' + str(first_sp_num) + '.mat')
            self.segs = segs['segmentmaps']
        if datasetname=='paviaU_':
            segs = sio.loadmat('Superpixels/'+'paviau_' + str(first_sp_num) + '.mat')
            self.segs = segs['segmentmaps']
        if datasetname == 'houston_wo_cloud_':
            segs = sio.loadmat('Superpixels/' + 'hst_' + str(first_sp_num) + '.mat')
            self.segs = segs['segmentmaps']    
            
        if datasetname == 'WHU_Hi_HongHu_':
            segs = sio.loadmat('Superpixels/' + 'WHU_Hi_HongHu_' + str(first_sp_num) + '.mat')
            self.segs = segs['segmentmaps']
        if datasetname == 'QUH_Tangdaowan_':
            segs = sio.loadmat('Superpixels/' + 'QUH_Tangdaowan_' + str(first_sp_num) + '.mat')
            self.segs = segs['segmentmaps']
        
        
    def getHierarchy(self):
        segs = self.segs
        layers, h, w = self.segs.shape
        segs = np.concatenate([np.reshape([i for i in range(h*w)], [1,h,w]), segs], axis=0)
        layers = layers+1
        S_list=[]

        for i in range(layers-1):
            S = np.zeros([np.max(segs[i])+1,np.max(segs[i+1])+1])
            l1 = np.reshape(segs[i],[-1])
            l2 = np.reshape(segs[i+1],[-1])
            for x in range(h*w):
                if S[l1[x] ,l2[x]]!=1: S[ l1[x] ,l2[x]]=1
            S_list.append(S)
        return S_list
