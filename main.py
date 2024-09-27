# coding=utf-8
import os
import cv2
import time
import torch
import argparse
import numpy as np
import scipy.io as sio
from sklearn import preprocessing

import psformer
from superhierarchy import SuperHierarchy
from utils import get_Samples_GT, GT_To_One_Hot, evaluate_performance, compute_loss
path = 'results/'

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--data', type=int, default=1) #dataset
parser.add_argument('--tr', type=int, default=30) #train_samples_per_class
## param
parser.add_argument('--l1', type=int, default=4) #stage number
parser.add_argument('--l2', type=int, default=3) #encoder layer number
parser.add_argument('--sp1', type=int, default=1024) #the first superpixel number
parser.add_argument('--cuda', type=str, default='0',
                    help="Specify CUDA device")

args = parser.parse_args()
if torch.cuda.is_available():
    print("Computation on CUDA GPU device {}".format(args.cuda))
    device = torch.device('cuda:{}'.format(args.cuda))
else:
    print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
    device = torch.device('cpu')

## [(1,30,4,3,1024), (2,30,4,2,1536), (3,30,4,1,2048), (4,100,4,3,2048), (5,30,4,5,1024)]
for (FLAG, curr_train_ratio, stage_num, encoder_num, first_sp_num) in \
    [(args.data, args.tr, args.l1, args.l2, args.sp1)]:

    print("FLAG, curr_train_ratio, stage_num, encoder_num, first_sp_num")
    print(FLAG, curr_train_ratio, stage_num, encoder_num, first_sp_num)
    torch.cuda.empty_cache()
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    Train_Time_ALL=[]
    Test_Time_ALL=[]
    samples_type = 'same_num'

    Seed_List = [111,222,333,444,555]

    if FLAG == 1:
        data_mat = sio.loadmat('/mnt/data/zjq/salinas/Salinas_corrected.mat')
        data = data_mat['salinas_corrected']
        gt_mat = sio.loadmat('/mnt/data/zjq/salinas/Salinas_gt.mat')
        gt = gt_mat['salinas_gt']
        
        class_count = 16  
        learning_rate = 5e-4  
        max_epoch = 500 
        dataset_name = "salinas_" 
 
    if FLAG == 2:
        data_mat = sio.loadmat('/mnt/data/zjq/paviaU/PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('/mnt/data/zjq/paviaU/PaviaU_gt.mat')
        gt = gt_mat['paviaU_gt']
 
        class_count = 9 
        learning_rate = 5e-4  
        max_epoch = 600 
        dataset_name = "paviaU_"  
 
    if FLAG == 3:
        data_mat = sio.loadmat('/mnt/data/zjq/Houston.mat')
        data = data_mat["input"]
        gt = data_mat["TR"]+data_mat["TE"]
        
        class_count = 15  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 300  # 迭代次数
        dataset_name = "houston_wo_cloud_"  # 数据集名称
        samples_type = 'disjoint'
    
    if FLAG == 4:
        data = sio.loadmat('/mnt/data/zjq/WHU-Hi-HongHu/WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
        gt = sio.loadmat('/mnt/data/zjq/WHU-Hi-HongHu/WHU_Hi_HongHu_gt.mat')['WHU_Hi_HongHu_gt']
        
        class_count = 22  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 600  # 迭代次数
        dataset_name = "WHU_Hi_HongHu_"

    if FLAG == 5:
        data = sio.loadmat('/mnt/data3/zjq/QUH_datasets_and_segmentmaps/QUH-Tangdaowan.mat')['Tangdaowan']
        gt = sio.loadmat('/mnt/data3/zjq/QUH_datasets_and_segmentmaps/QUH-Tangdaowan_GT.mat')['TangdaowanGT']

        val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 18  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 600  # 迭代次数
        dataset_name = "QUH_Tangdaowan_" 


    for curr_seed in Seed_List:
        train_samples_gt, test_samples_gt = get_Samples_GT(curr_seed, gt, class_count, curr_train_ratio, samples_type)
        ## 如果使用“Single Scale/4”或者“Single Scale/32”，将数据预先padding成能被4或者32整除的大小，借鉴郑卓师兄的simplecv
        # data, train_samples_gt, test_samples_gt = preset(data, train_samples_gt, test_samples_gt)

        Test_GT = test_samples_gt
        m, n, d = data.shape  
        height, width, bands = data.shape  
        ## gt → onehot
        train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
        test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
        train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
        test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)  
        ## gt → mask
        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m* n, class_count])
        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m* n, class_count])
        ## HSI data归一化
        data = np.reshape(data, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        data = np.reshape(data, [height, width, bands])

        ############# core code #############
        tic = time.time()
        SM = SuperHierarchy(dataset_name, first_sp_num)
        S_list = SM.getHierarchy()
        S_list = S_list[0:int(stage_num)] ## Q matrix
        ## (21025, 2048) (2048, 1024) (1024, 512) (512, 256)
        
        toc = time.time()
        HierarchicalSegmentation_Time = toc - tic
        print('getHierarchy -- cost time:', toc - tic)
        
        ##### 将所有数据转成GPU
        S_list_gpu=[]
        for i in range(len(S_list)):
            S_list_gpu.append(torch.from_numpy(np.array(S_list[i],dtype=np.float32)).to(device))
            
        train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)

        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)

        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)

        net_input=torch.from_numpy(data.astype(np.float32)).to(device)
        net = psformer.PSFormer(bands, class_count, S_list_gpu, stage_num, encoder_num)

        # ## 统计计算复杂度
        # from torchsummary import summary
        # from fvcore.nn import FlopCountAnalysis
        # from thop import profile
        # from thop import clever_format
        # print(net_input.shape,"net_input")
        # input = net_input
        # # summary(net.to(device), input.size())
        # macs, params = profile(net.to(device), inputs=(input.to(device), ))
        # macs, params = clever_format([macs, params], "%.6f")
        # print("macs, params", macs, params) 

        # # flops = FlopCountAnalysis(net.to(device), input.to(device))
        # # print("flops.total()",flops.total())
        # exit()

        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate) #,weight_decay=0.0001

        ## train the network
        net.train()
        tic1 = time.time()
        for i in range(max_epoch+1):
            optimizer.zero_grad()  # zero the gradient buffers
            output, _ = net(net_input)
            loss = compute_loss(output,train_samples_gt_onehot,train_label_mask, m, n)
            loss.backward(retain_graph=False)
            optimizer.step()  # Does the update
            if i%50==0:
                with torch.no_grad():
                    net.eval()
                    output, _ = net(net_input)
                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask, m, n)
                    trainOA = evaluate_performance(path, OA_ALL, AA_ALL, KPP_ALL, AVG_ALL, device, dataset_name, output, train_samples_gt, train_samples_gt_onehot, m, n, class_count, Test_GT, curr_train_ratio, stage_num, encoder_num, first_sp_num)
                    valloss = compute_loss(output, test_samples_gt_onehot, test_label_mask, m, n)
                    valOA = evaluate_performance(path, OA_ALL, AA_ALL, KPP_ALL, AVG_ALL, device, dataset_name, output, test_samples_gt, test_samples_gt_onehot, m, n, class_count, Test_GT, curr_train_ratio, stage_num, encoder_num, first_sp_num)
                    print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA, valloss, valOA))
                    torch.save(net.state_dict(),"model/{}best_model.pt".format(dataset_name))
                        
                torch.cuda.empty_cache()
                net.train()
        toc1 = time.time()
        print("\n\n====================training done. starting evaluation...========================\n")
        training_time=toc1 - tic1 + HierarchicalSegmentation_Time 
        Train_Time_ALL.append(training_time)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model/{}best_model.pt".format(dataset_name)))
            net.eval()
            tic2 = time.time()
            output, vis_attn = net(net_input)
            toc2 = time.time()
            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask, m, n)
            testOA, OA_ALL, AA_ALL, KPP_ALL, AVG_ALL = evaluate_performance(path, OA_ALL, AA_ALL, KPP_ALL, AVG_ALL, device, dataset_name, output, test_samples_gt, test_samples_gt_onehot, m, n, class_count, Test_GT, curr_train_ratio, stage_num, encoder_num, first_sp_num, require_AA_KPP=True,printFlag=False)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
            #
            classification_map=torch.argmax(output, 1).reshape([height,width]).cpu()+1
            pred_mat=classification_map.data.numpy()
            sio.savemat("{:.02f}_{}PSFormer_pred_mat.mat".format(testOA*100, dataset_name),{"pred_mat":pred_mat})
            testing_time=toc2 - tic2 + HierarchicalSegmentation_Time 
            Test_Time_ALL.append(testing_time)
            
            ## visualization analysis
            # savedir =  'vis/{}'.format(dataset_name)         
            # for idx in range(5):   
            #     x_visualize = vis_attn[idx]
            #     print(np.min(x_visualize),np.max(x_visualize),"minmax")
            #     x_visualize = (x_visualize*255).astype(np.uint8)
            #     # (((x_visualize - np.min(x_visualize))/(np.max(x_visualize)-np.min(x_visualize)))*255).astype(np.uint8) #归一化并映射到0-255的整数，方便伪彩色化
            #     x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理   
            #     cv2.imwrite(savedir+str(idx+1)+'.jpg',x_visualize) #保存可视化图像
        torch.cuda.empty_cache()
        del net
            
    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL=np.array(Train_Time_ALL)
    Test_Time_ALL=np.array(Test_Time_ALL)

    print("OA List: ", OA_ALL)
    print("\nTrain ratio={}, Stage={}, Encoder Layer={}, First SP={}".format(curr_train_ratio, stage_num, encoder_num, first_sp_num),
          "\n==============================================================================")
    print('OA=', np.mean(OA_ALL), '±', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '±', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '±', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '±', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
        
    # save information 存储所有种子的平均结果
    f = open(path + dataset_name + 'results_tr{}_stage{}_encoder{}_firstsp{}.txt'.format(curr_train_ratio, stage_num, encoder_num, first_sp_num), 'a+')
    str_results = '\n\n************************************************' \
    +"\nTrain ratio={}, Stage={}, Encoder Layer={}, First SP={}".format(curr_train_ratio, stage_num, encoder_num, first_sp_num) \
    +'\nAVG='+ str(['{:.2f}±{:.02f}'.format((np.mean(AVG_ALL,0)*100)[idx], (np.std(AVG_ALL,0)*100)[idx]) for idx in range(class_count)]) \
    +'\nOA={:.02f}±{:.02f}'.format(np.mean(OA_ALL)*100, np.std(OA_ALL)*100) \
    +'\nAA={:.02f}±{:.02f}'.format(np.mean(AA_ALL)*100, np.std(AA_ALL)*100) \
    +'\nKpp={:.02f}±{:.02f}'.format(np.mean(KPP_ALL)*100, np.std(KPP_ALL)*100) \
    +"\nAverage training time:{:.02f}".format(np.mean(Train_Time_ALL)) \
    +"\nAverage testing time:{:.02f}".format(np.mean(Test_Time_ALL))
    f.write(str_results)
    f.close()
        

    
    
    
    
    
    
    
