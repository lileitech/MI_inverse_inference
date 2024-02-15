import os
import random
import numpy as np
import torch
import glob
import torch.utils.data as data
import sys
import pyvista
sys.path.append('.')
sys.path.append('..')
from utils import visualize_PC_with_label
import re

class LoadDataset(data.Dataset):
    def __init__(self, path, num_input=2048, split='train'): #16384
        self.path = path
        self.num_input = num_input
        self.use_cobiveco = True
        self.data_augment = False
        self.signal_length = 512

        with open(path + 'my_split/{}.list'.format(split), 'r') as f:
            filenames = [line.strip() for line in f]

        self.metadata = list()
        for filename in filenames:
            print(filename)
            datapath = path + filename + '/'

            unit = 0.1
            if self.use_cobiveco:
                nodesXYZ, label_index = getCobiveco_vtu(datapath + filename + '_cobiveco_AHA17.vtu')
            else:
                nodesXYZ = np.loadtxt(datapath + filename + '_xyz.csv', delimiter=',')
                label_index = np.zeros((nodesXYZ.shape[0], 1))
                LVendo_node = np.unique((np.loadtxt(datapath + filename + '_lvface.csv', delimiter=',')-1).astype(int))
                RVendo_node = np.unique((np.loadtxt(datapath + filename + '_rvface.csv', delimiter=',')-1).astype(int))
                epi_node = np.unique((np.loadtxt(datapath + filename + '_epiface.csv', delimiter=',')-1).astype(int))
                label_index[LVendo_node] = 1
                label_index[RVendo_node] = 2
                label_index[epi_node] = 3
                label_index = label_index[..., np.newaxis]
                surface_index = np.concatenate((LVendo_node, RVendo_node, epi_node), axis=0) 
            
            PC_XYZ_labeled = np.concatenate((unit*nodesXYZ, label_index), axis=1)           
            electrode_node = np.loadtxt(datapath + filename + '_electrodePositions.csv', delimiter=',')
            Coord_base_apex = np.loadtxt(datapath + filename + '_BaseApexCoord.csv', delimiter=',')
            Coord_apex, Coord_base = Coord_base_apex[1], Coord_base_apex[0] 
            electrode_index = 4*np.ones(electrode_node.shape[0], dtype=np.int32)
            electrode_XYZ_labeled = np.concatenate((unit*electrode_node, electrode_index[..., np.newaxis]), axis=1)
            
            signal_files = glob.glob(datapath + filename + '*_simulated_ECG' + '*.csv')
            num_signal = len(signal_files)
            # print(num_signal)
            for id in range(num_signal):
                MI_index = np.zeros(nodesXYZ.shape[0], dtype=np.int32)
                ECG_value = np.loadtxt(signal_files[id], delimiter=',')
                ECG_value_u = np.pad(ECG_value, ((0, 0), (0, self.signal_length-ECG_value.shape[1])), 'constant')           
                MI_type = signal_files[id].replace(path, '').replace(filename, '').replace('_simulated_ECG_', '').replace('.csv', '').replace('\\', '')
                
                if MI_type == 'B1_large_transmural_slow' or MI_type == 'normal' or MI_type == 'A2_30_40_transmural':
                    continue

                if re.compile(r'5_transmural|0_transmural', re.IGNORECASE).search(MI_type): # remove apical MI size test case
                    continue

                if re.compile(r'AHA', re.IGNORECASE).search(MI_type): # remove randomly generated MI
                    continue

                # if not re.compile(r'5_transmural|0_transmural', re.IGNORECASE).search(MI_type) and not (MI_type == 'A2_transmural'): # remove apical MI size test case
                #     continue

                # if not re.compile(r'AHA', re.IGNORECASE).search(MI_type): # test only random MI!
                #     continue
                #             
                # if MI_type.find('subendo') != -1:
                #     continue
 
                # if MI_type != 'B3_transmural' and MI_type != 'A3_transmural' and MI_type != 'A2_transmural':
                #     continue  

                # print(MI_type)             

                if MI_type != 'normal':
                    Scar_filename = signal_files[id].replace('simulated_ECG', 'lvscarnodes')
                    BZ_filename = signal_files[id].replace('simulated_ECG', 'lvborderzonenodes')
                    if MI_type == 'B1_large_transmural_slow':
                        Scar_filename = Scar_filename.replace('_slow', '')
                        BZ_filename = BZ_filename.replace('_slow', '')

                    Scar_node = np.unique((np.loadtxt(Scar_filename, delimiter=',')-1).astype(int))
                    BZ_node = np.unique((np.loadtxt(BZ_filename, delimiter=',')-1).astype(int))
                    MI_index[Scar_node] = 1
                    MI_index[BZ_node] = 2
                ECG_array = np.array(ECG_value_u)
                MI_array = np.array(MI_index)
                MI_type_id = np.array(id)
                # print(MI_type_id)
                
                partial_PC_labeled_array, idx_remained = resample_pcd(PC_XYZ_labeled, self.num_input)
                partial_MI_lab_array = MI_array[idx_remained]
                partial_PC_labeled_array_coarse, idx_remained = resample_pcd(PC_XYZ_labeled, self.num_input//4)             
                # visualize_PC_with_label(partial_PC_labeled_array[:, 0:3], partial_MI_array)
                partial_PC_electrode_labeled_array = partial_PC_labeled_array # np.concatenate((partial_PC_labeled_array, electrode_XYZ_labeled), axis=0)
                partial_PC_electrode_XYZ = partial_PC_electrode_labeled_array[:, 0:3]
                partial_PC_electrode_lab = partial_PC_electrode_labeled_array[:, 3:]
                # partial_MI_lab_array = partial_MI_lab_array + np.where(partial_PC_electrode_labeled_array[0:self.num_input, -1]==1.0, 3, 0)
                # visualize_PC_with_label(partial_PC_labeled_array[:, 0:3], partial_MI_lab_array)

                partial_PC_electrode_XYZ_normalized = normalize_data(partial_PC_electrode_XYZ, Coord_apex)
                if self.data_augment:
                    scaling = random.uniform(0.8, 1.2)
                    partial_PC_electrode_XYZ_normalized = scaling*translate_point(jitter_point(rotate_point(partial_PC_electrode_XYZ_normalized, np.random.random()*np.pi)))
                partial_PC_electrode_XYZ_normalized_labeled = np.concatenate((partial_PC_electrode_XYZ_normalized, partial_PC_electrode_lab), axis=1)

                partial_PC_electrode_XYZ_normalized_coarse = normalize_data(partial_PC_labeled_array_coarse[:, 0:3], Coord_apex)
                partial_PC_electrode_XYZ_normalized_labeled_coarse = np.concatenate((partial_PC_electrode_XYZ_normalized_coarse, partial_PC_labeled_array_coarse[:, 3:]), axis=1)

                self.metadata.append((partial_PC_electrode_XYZ_normalized_labeled, partial_MI_lab_array, ECG_array, partial_PC_electrode_XYZ_normalized_labeled_coarse, MI_type))

    def __getitem__(self, index):
        partial_PC_electrode_XYZ_normalized_labeled, partial_MI_array, ECG_array, partial_PC_electrode_XYZ, MI_type = self.metadata[index]

        partial_input = torch.from_numpy(partial_PC_electrode_XYZ_normalized_labeled).float()
        gt_MI = torch.from_numpy(partial_MI_array).long()
        ECG_input = torch.from_numpy(ECG_array).float()
        partial_input_coarse = torch.from_numpy(partial_PC_electrode_XYZ).float()

        return partial_input, ECG_input, gt_MI, partial_input_coarse, MI_type

    def __len__(self):
        return len(self.metadata)


class LoadDataset_all(data.Dataset):
    def __init__(self, path, num_input=2048, split='train'): #16384
        self.path = path
        self.num_input = num_input
        self.use_cobiveco = False
        self.data_augment = False
        self.signal_length = 512

        with open(path + 'my_split/{}.list'.format(split), 'r') as f:
            filenames = [line.strip() for line in f]


        self.metadata = list()
        for filename in filenames:
            print(filename)
            datapath = path + filename + '/'

            unit = 1
            if self.use_cobiveco:
                nodesXYZ, label_index = getCobiveco_vtu(datapath + filename + '_heart_cobiveco.vtu')
            else:
                nodesXYZ = np.loadtxt(datapath + filename + '_xyz.csv', delimiter=',')
                label_index = np.zeros((nodesXYZ.shape[0], 1), dtype=np.int)
                LVendo_node = np.unique((np.loadtxt(datapath + filename + '_lvface.csv', delimiter=',')-1).astype(int))
                RVendo_node = np.unique((np.loadtxt(datapath + filename + '_rvface.csv', delimiter=',')-1).astype(int))
                epi_node = np.unique((np.loadtxt(datapath + filename + '_epiface.csv', delimiter=',')-1).astype(int))
                label_index[LVendo_node] = 1
                label_index[RVendo_node] = 2
                label_index[epi_node] = 3
                surface_index = np.concatenate((LVendo_node, RVendo_node, epi_node), axis=0) 
            
            PC_XYZ_labeled = np.concatenate((unit*nodesXYZ, label_index), axis=1)           
            electrode_node = np.loadtxt(datapath + filename + '_electrodePositions.csv', delimiter=',')
            Coord_base_apex = np.loadtxt(datapath + filename + '_BaseApexCoord.csv', delimiter=',')
            Coord_apex, Coord_base = Coord_base_apex[1], Coord_base_apex[0] 
            electrode_index = 4*np.ones(electrode_node.shape[0], dtype=np.int)
            electrode_XYZ_labeled = np.concatenate((unit*electrode_node, electrode_index[..., np.newaxis]), axis=1)
            
            signal_files = glob.glob(datapath + filename + '*_simulated_ECG' + '*.csv')
            ECG_list, MI_index_list = list(), list()
            MItype_list = list()
            num_signal = len(signal_files)
            for id in range(num_signal):
                MI_index = np.zeros(nodesXYZ.shape[0], dtype=np.int)
                ECG_value = np.loadtxt(signal_files[id], delimiter=',')
                ECG_value_u = np.pad(ECG_value, ((0, 0), (0, self.signal_length-ECG_value.shape[1])), 'constant')           
                MI_type = signal_files[id].replace(path, '').replace(filename, '').replace('_simulated_ECG_', '').replace('.csv', '').replace('\\', '')
                if MI_type == 'B1_large_transmural_slow' or MI_type == 'B1_large_transmural_slow':
                    continue
                if MI_type != 'normal':
                    Scar_filename = signal_files[id].replace('simulated_ECG', 'lvscarnodes')
                    BZ_filename = signal_files[id].replace('simulated_ECG', 'lvborderzonenodes')
                    Scar_node = np.unique((np.loadtxt(Scar_filename, delimiter=',')-1).astype(int))
                    BZ_node = np.unique((np.loadtxt(BZ_filename, delimiter=',')-1).astype(int))
                    MI_index[Scar_node] = 421
                    MI_index[BZ_node] = 422

                ECG_list.append(ECG_value_u)
                MI_index_list.append(MI_index)
                MItype_list.append(MI_type)
                                                           
            ECG_array = np.array(ECG_list).transpose(1, 2, 0) 
            MI_array = np.array(MI_index_list).transpose(1, 0)                     
            partial_PC_labeled_array, idx_remained = resample_pcd(PC_XYZ_labeled[surface_index], self.num_input)
            partial_MI_array = MI_array[surface_index][idx_remained]
            partial_PC_electrode_labeled_array = np.concatenate((partial_PC_labeled_array, electrode_XYZ_labeled), axis=0)
            partial_PC_electrode_XYZ = partial_PC_electrode_labeled_array[:, 0:3]
            partial_PC_electrode_lab = np.expand_dims(partial_PC_electrode_labeled_array[:, 3], axis=1)
            partial_PC_electrode_XYZ_normalized = normalize_data(partial_PC_electrode_XYZ, Coord_apex)
            if self.data_augment:
                scaling = random.uniform(0.8, 1.2)
                partial_PC_electrode_XYZ_normalized = scaling*translate_point(jitter_point(rotate_point(partial_PC_electrode_XYZ_normalized, np.random.random()*np.pi)))
            partial_PC_electrode_XYZ_normalized_labeled = np.concatenate((partial_PC_electrode_XYZ_normalized, partial_PC_electrode_lab), axis=1)
          
                      
            self.metadata.append((partial_PC_electrode_XYZ_normalized_labeled, partial_MI_array, ECG_array, partial_PC_electrode_XYZ))

    def __getitem__(self, index):
        partial_PC_electrode_XYZ_normalized_labeled, partial_MI_array, ECG_array, partial_PC_electrode_XYZ = self.metadata[index]

        ECG_array[np.isnan(ECG_array)] = 0  # ECG output with a size of [n_batch, 8*256], covert the nan value into 0
        partial_input = torch.from_numpy(partial_PC_electrode_XYZ_normalized_labeled).float()
        gt_MI, ECG_input = torch.from_numpy(partial_MI_array).float(), torch.from_numpy(ECG_array).float()
        partial_input_ori = torch.from_numpy(partial_PC_electrode_XYZ).float()

        return partial_input, ECG_input, gt_MI, partial_input_ori

    def __len__(self):
        return len(self.metadata)


def getCobiveco_vtu(cobiveco_fileName): # Read Cobiveco data in .vtu format (added by Lei on 2023/01/30)
    cobiveco_vol = pyvista.read(cobiveco_fileName) #, force_ext='.vtu'

    cobiveco_nodesXYZ = cobiveco_vol.points
    cobiveco_nodes_array = cobiveco_vol.point_data
    # Apex-to-Base - ab
    ab = cobiveco_nodes_array['ab']
    # Rotation angle - rt
    rt = cobiveco_nodes_array['rt']
    # Transmurality - tm
    tm = cobiveco_nodes_array['tm']
    # Ventricle - tv
    tv = cobiveco_nodes_array['tv']
    # AHA-17 map - aha
    aha = cobiveco_nodes_array['aha']

    return cobiveco_nodesXYZ, np.transpose(np.array([ab, rt, tm, tv, aha], dtype=float))

###　point cloud augmentation　###
# translate point cloud
def translate_point(point):
    point = np.array(point)
    shift = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
    shift = np.expand_dims(np.array(shift), axis=0)
    shifted_point = np.repeat(shift, point.shape[0], axis=0)
    shifted_point += point

    return shifted_point

# add Gaussian noise 
def jitter_point(point, sigma=0.01, clip=0.01):
    assert(clip > 0)
    point = np.array(point)
    point = point.reshape(-1,3)
    Row, Col = point.shape
    jittered_point = np.clip(sigma * np.random.randn(Row, Col), -1*clip, clip)
    jittered_point += point

    return jittered_point

# rotate point cloud
def rotate_point(point, rotation_angle=0.5*np.pi):
    point = np.array(point)
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    # Rotation around X axis
    rotation_matrix_X = np.array([[1, 0, 0],
                                [0, cos_theta, -sin_theta],
                                [0, sin_theta, cos_theta]])
    # Rotation around Y axis
    rotation_matrix_Y = np.array([[cos_theta, 0, sin_theta],
                                [0, 1, 0],
                                [-sin_theta, 0, cos_theta]])
    # Rotation around Z axis
    rotation_matrix_Z = np.array([[cos_theta, sin_theta, 0],
                                [-sin_theta, cos_theta, 0],
                                [0, 0, 1]])   

    rotated_point = np.dot(point.reshape(-1, 3), rotation_matrix_Z)

    return rotated_point

# normalize point cloud based on apex coordinate
def normalize_data(PC, Coord_apex):
    """ Normalize the point cloud, use coordinates of centroid/ apex,
        Input:
            NxC array
        Output:
            NxC array
    """
    N, C = PC.shape
    normal_data = np.zeros((N, C))
    # centroid = np.mean(PC, axis=0)
    PC = PC - Coord_apex
    # m = np.max(np.sqrt(np.sum(PC ** 2, axis=1)))
    # PC = PC / m
    # normal_data = PC

    # compute the minimum and maximum values of each coordinate
    min_coords = np.min(PC, axis=0)
    max_coords = np.max(PC, axis=0)

    # normalize the point cloud coordinates
    normal_data = (PC - min_coords) / (max_coords - min_coords)

    return normal_data

def resample_pcd_ATM(pcd, ATM, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx_root_nodes = np.where(ATM[:, 0]==1.0) # ATM[:, 0]
    prob = 1/(pcd.shape[0]-idx_root_nodes[0].shape[0])
    node_prob = prob*np.ones(pcd.shape[0])
    node_prob[idx_root_nodes] = 0
    idx = np.random.choice(np.arange(pcd.shape[0]), n-idx_root_nodes[0].shape[0], p=node_prob, replace=False)
    idx_remained = np.union1d(idx, idx_root_nodes)
    # idx_updated_permuted = np.random.permutation(idx_updated)
    # if idx_updated_permuted.shape[0] < n:
    #     idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    
    return pcd[idx_remained], ATM[idx_remained], idx_remained

def resample_pcd_ATM_ori(pcd, ATM, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]], ATM[idx[:n]]

def resample_gd(gt_output, num_coarse, num_dense): #added by Lei in 2022/02/10 to seperately resample groundtruth label
    """Drop or duplicate points so that pcd has exactly n points"""
    choice = np.random.choice(len(gt_output), num_coarse, replace=True)
    coarse_gt = gt_output[choice, :]
    dense_gt = resample_pcd(gt_output, num_dense)
    return coarse_gt, dense_gt

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]], idx[:n]


if __name__ == '__main__':
    ROOT = './dataset/'
    GT_ROOT = os.path.join(ROOT, 'gt')
    PARTIAL_ROOT = os.path.join(ROOT, 'partial')

    train_dataset = LoadDataset(partial_path=PARTIAL_ROOT, gt_path=GT_ROOT, split='train')
    val_dataset = LoadDataset(partial_path=PARTIAL_ROOT, gt_path=GT_ROOT, split='val')
    test_dataset = LoadDataset(partial_path=PARTIAL_ROOT, gt_path=GT_ROOT, split='test')
    print("\033[33mTraining dataset\033[0m has {} pair of partial and ground truth point clouds".format(len(train_dataset)))
    print("\033[33mValidation dataset\033[0m has {} pair of partial and ground truth point clouds".format(len(val_dataset)))
    print("\033[33mTesting dataset\033[0m has {} pair of partial and ground truth point clouds".format(len(test_dataset)))

    # visualization
    input_pc, coarse_pc, dense_pc, conditions = train_dataset[random.randint(0, len(train_dataset))-1]
    print("partial input point cloud has {} points".format(len(input_pc)))
    print("coarse output point cloud has {} points".format(len(coarse_pc)))
    print("dense output point cloud has {} points".format(len(dense_pc)))
