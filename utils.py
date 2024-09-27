import torch
import random
import numpy as np
import scipy.io as sio
from sklearn import metrics

######### related to Main.py #########

def evaluate_performance(path, OA_ALL, AA_ALL, KPP_ALL, AVG_ALL, device, dataset_name, network_output,train_samples_gt,train_samples_gt_onehot, m, n, class_count, Test_GT, curr_train_ratio, stage_num, encoder_num, first_sp_num, require_AA_KPP=False,printFlag=True):
    zeros = torch.zeros([m * n]).to(device).float()
    if False==require_AA_KPP:
        with torch.no_grad():
            available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
            available_label_count=available_label_idx.sum()#有效标签的个数
            correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
            OA= correct_prediction.cpu()/available_label_count
            
            return OA
    else:
        with torch.no_grad():
            #计算OA
            available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
            available_label_count=available_label_idx.sum()#有效标签的个数
            correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
            OA= correct_prediction.cpu()/available_label_count
            OA=OA.cpu().numpy()
            
            # 计算AA
            zero_vector = np.zeros([class_count])
            output_data=network_output.cpu().numpy()
            train_samples_gt=train_samples_gt.cpu().numpy()
            train_samples_gt_onehot=train_samples_gt_onehot.cpu().numpy()
            
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            for z in range(output_data.shape[0]):
                if ~(zero_vector == output_data[z]).all():
                    idx[z] += 1
            
            count_perclass = np.zeros([class_count])
            correct_perclass = np.zeros([class_count])
            for x in range(len(train_samples_gt)):
                if train_samples_gt[x] != 0:
                    count_perclass[int(train_samples_gt[x] - 1)] += 1
                    if train_samples_gt[x] == idx[x]:
                        correct_perclass[int(train_samples_gt[x] - 1)] += 1
            test_AC_list = correct_perclass / count_perclass
            test_AA = np.average(test_AC_list)

            # 计算KPP
            test_pre_label_list = []
            test_real_label_list = []
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            idx = np.reshape(idx, [m, n])
            for ii in range(m):
                for jj in range(n):
                    if Test_GT[ii][jj] != 0:
                        test_pre_label_list.append(idx[ii][jj] + 1)
                        test_real_label_list.append(Test_GT[ii][jj])
            test_pre_label_list = np.array(test_pre_label_list)
            test_real_label_list = np.array(test_real_label_list)
            kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                              test_real_label_list.astype(np.int16))
            test_kpp = kappa

            # 输出
            if printFlag:
                print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                print('acc per class:')
                print(test_AC_list)

            OA_ALL.append(OA)
            AA_ALL.append(test_AA)
            KPP_ALL.append(test_kpp)
            AVG_ALL.append(test_AC_list)

            # save
            f = open(path + dataset_name + 'results_tr{}_stage{}_encoder{}_firstsp{}.txt'.format(curr_train_ratio, stage_num, encoder_num, first_sp_num), 'a+')
            str_results = "\nOA=" + str(OA) \
                          + "\nAA=" + str(test_AA) \
                          + '\nkpp=' + str(test_kpp) \
                          + '\nacc per class:' + str(test_AC_list) + "\n"
            f.write(str_results)
            f.close()
           
            return OA,OA_ALL,AA_ALL,KPP_ALL,AVG_ALL

def GT_To_One_Hot(gt, class_count):
    '''
    Convet Gt to one-hot labels
    :param gt: 2D
    :param class_count:
    :return:
    '''
    GT_One_Hot = []  # 转化为one-hot形式的标签
    [height, width]=gt.shape
    for i in range(height):
        for j in range(width):
            temp = np.zeros(class_count, dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot


# def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor, m, n):
#     real_labels = reallabel_onehot
#     we = -torch.mul(real_labels,torch.log(predict+1e-15))
#     we = torch.mul(we, reallabel_mask)

#     we2 = torch.sum(real_labels, 0)  
#     we2 = 1. / (we2+ 1)  
#     we2 = torch.unsqueeze(we2, 0)
#     we2 = we2.repeat([m * n, 1])
#     we = torch.mul(we, we2)
#     pool_cross_entropy = torch.sum(we)
#     return pool_cross_entropy

def compute_loss(predict, reallabel_onehot, reallabel_mask, m, n):
    real_labels = reallabel_onehot #[N,16]
    we = -torch.mul(real_labels,torch.log(predict))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy

def get_Samples_GT(seed, gt, class_count, train_ratio, samples_type='same_num'):
    # step2:随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
    random.seed(seed)
    [height, width] = gt.shape
    gt_reshape = np.reshape(gt, [-1])
    train_rand_idx = []
    val_rand_idx = []
    if samples_type == 'ratio':
        val_ratio = 0.05 ### need revise
        train_number_per_class=[]
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            rand_idx = random.sample(rand_list,
                                     np.ceil(samplesCount * train_ratio).astype('int32')+\
                                     np.ceil(samplesCount*val_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
            train_number_per_class.append(np.ceil(samplesCount * train_ratio).astype('int32'))
            rand_real_idx_per_class = idx[rand_idx]
            train_rand_idx.append(rand_real_idx_per_class)
        train_rand_idx = np.array(train_rand_idx)
        train_data_index = []
        val_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = list(train_rand_idx[c])
            train_data_index=train_data_index+a[:train_number_per_class[c]]
            val_data_index=val_data_index+a[train_number_per_class[c]:]
        
        ##将测试集（所有样本，包括训练样本）也转化为特定形式
        train_data_index = set(train_data_index)
        val_data_index=set(val_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)
        
        # 背景像元的标签
        test_data_index = all_data_index - train_data_index - val_data_index
        
        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)
    
    if samples_type == 'same_num':
        print("random sample selection strategy！！！")
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            real_train_samples_per_class = int(train_ratio) # 每类相同数量样本,则训练比例为每类样本数量
            
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            if real_train_samples_per_class >= samplesCount:
                real_train_samples_per_class = 15 #samplesCount-1
            
            rand_idx = random.sample(rand_list,
                                     real_train_samples_per_class)
            rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
            train_rand_idx.append(rand_real_idx_per_class_train)

        train_rand_idx = np.array(train_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)

        train_data_index = set(train_data_index)
        
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)
        
        # 背景像元的标签
        test_data_index = all_data_index - train_data_index
        
        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        
        # 获取训练样本的标签图
        train_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(train_data_index)):
            train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        
        # 获取测试样本的标签图
        test_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(test_data_index)):
            test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]

        train_samples_gt = np.reshape(train_samples_gt, [height, width])
        test_samples_gt = np.reshape(test_samples_gt, [height, width])

    if samples_type == 'disjoint':
        print("disjoint sample selection strategy！！！")
        train_samples_gt = sio.loadmat('/mnt/data/zjq/Houston.mat')["TR"]
        test_samples_gt = sio.loadmat('/mnt/data/zjq/Houston.mat')["TE"]

    return train_samples_gt, test_samples_gt