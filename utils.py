import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
EPS = 1e-4

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar):
        var = torch.exp(logvar) + EPS
        T = 1. / (var + EPS)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + EPS)

        return pd_mu, pd_logvar

class alphaProductOfExperts(nn.Module):
    """Return parameters for weighted product of independent experts (mmJSD implementation).
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar, weights=None):
        if weights is None:
            num_components = mu.shape[0]
            weights = (1/num_components) * torch.ones(mu.shape).to(mu.device)
    
        var = torch.exp(logvar) + EPS
        T = 1. / (var + EPS)
        weights = torch.broadcast_to(weights, mu.shape)
        pd_var = 1. / torch.sum(weights * T + EPS, dim=0)
        pd_mu = pd_var * torch.sum(weights * mu * T, dim=0)
        pd_logvar = torch.log(pd_var + EPS)
        
        return pd_mu, pd_logvar
    
class weightedProductOfExperts(nn.Module):
    """Return parameters for weighted product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar, weight):

        var = torch.exp(logvar) + EPS     
        weight = weight[:, None, :].repeat(1, mu.shape[1],1)
        T = 1.0 / (var + EPS)
        pd_var = 1. / torch.sum(weight * T + EPS, dim=0)
        pd_mu = pd_var * torch.sum(weight * mu * T, dim=0)
        pd_logvar = torch.log(pd_var + EPS)
        return pd_mu, pd_logvar

class MixtureOfExperts(nn.Module):
    """Return parameters for mixture of independent experts.
    Implementation from: https://github.com/thomassutter/MoPoE

    Args:
    mus (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvars (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mus, logvars):

        num_components = mus.shape[0]
        num_samples = mus.shape[1]
        weights = (1/num_components) * torch.ones(num_components).to(mus[0].device)
        idx_start = []
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0
            else:
                i_start = int(idx_end[k-1])
            if k == num_components-1:
                i_end = num_samples
            else:
                i_end = i_start + int(torch.floor(num_samples*weights[k]))
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples

        mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])
        logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])

        return mu_sel, logvar_sel

class MeanRepresentation(nn.Module):
    """Return mean of separate VAE representations.
    
    Args:
    mu (torch.Tensor): Mean of distributions. M x D for M views.
    logvar (torch.Tensor): Log of Variance of distributions. M x D for M views.
    """

    def forward(self, mu, logvar):
        mean_mu = torch.mean(mu, axis=0)
        mean_logvar = torch.mean(logvar, axis=0)
        
        return mean_mu, mean_logvar


def visualize_PC_with_twolabel_rotated(nodes_xyz_pre, labels_pre, labels_gd, filename='PC_label.pdf'):
    # Define custom colors for labels
    color_dict = {0: '#BCB6AE', 1: '#288596', 2: '#7D9083'}

    df = pd.DataFrame(nodes_xyz_pre, columns=['x', 'y', 'z'])
    colors_gd = [color_dict[label] for label in labels_gd]
    colors_pre = [color_dict[label] for label in labels_pre]
    

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    ax1.scatter(df['x'], df['y'], df['z'], c=colors_gd, s=1.5)  
    ax1.set_title('Ground truth')
    ax2.scatter(df['x'], df['y'], df['z'], c=colors_pre, s=1.5) 
    ax2.set_title('Prediction')
    ax1.set_axis_off() # Hide coordinate space 
    ax2.set_axis_off() # Hide coordinate space

    # 定义交互事件函数
    def on_rotate(event):
        # 获取当前旋转的角度
        elev = ax1.elev
        azim = ax1.azim
        
        # 设置两个子图的视角
        ax1.view_init(elev=elev, azim=azim)
        ax2.view_init(elev=elev, azim=azim)
        
        # 更新图形
        fig.canvas.draw()

    # 绑定交互事件
    fig.canvas.mpl_connect('motion_notify_event', on_rotate)

    plt.show()

def visualize_PC_with_twolabel(nodes_xyz_pre, labels_pre, labels_gd, filename='PC_label.pdf'):
    # Define custom colors for labels
    color_dict = {0: '#BCB6AE', 1: '#288596', 2: '#7D9083'}

    df = pd.DataFrame(nodes_xyz_pre, columns=['x', 'y', 'z'])
    colors_pre = [color_dict[label] for label in labels_pre]
    colors_gd = [color_dict[label] for label in labels_gd]

    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.scatter(df['x'], df['y'], df['z'], c=colors_pre, s=1.5)  
    ax1.set_axis_off() # Hide coordinate space
    ax2 = fig.add_subplot(121, projection='3d')
    ax2.scatter(df['x'], df['y'], df['z'], c=colors_gd, s=1.5)    
    ax2.set_axis_off() # Hide coordinate space
    plt.subplots_adjust(wspace=0)
    plt.savefig(filename)
    # plt.show()
    plt.close(fig)

def visualize_two_PC(nodes_xyz_pre, nodes_xyz_gd, labels, filename='PC_recon.pdf'):
    color_dict = {0: '#BCB6AE', 1: '#BCB6AE', 2: '#BCB6AE'}
    colors = [color_dict[label] for label in labels]

    df_pre = pd.DataFrame(nodes_xyz_pre, columns=['x', 'y', 'z'])
    df_gd = pd.DataFrame(nodes_xyz_gd, columns=['x', 'y', 'z'])

    fig = plt.figure(figsize=(4, 6))
    ax1 = fig.add_subplot(212, projection='3d')
    ax1.scatter(df_pre['x'], df_pre['y'], df_pre['z'], c=colors, s=1.5)  
    ax1.set_axis_off() # Hide coordinate space
    ax2 = fig.add_subplot(211, projection='3d')
    ax2.scatter(df_gd['x'], df_gd['y'], df_gd['z'], c=colors, s=1.5)    
    ax2.set_axis_off() # Hide coordinate space
    plt.subplots_adjust(hspace=0)
    plt.savefig(filename)
    # plt.show()
    plt.close(fig)

def visualize_PC_with_label(nodes_xyz, labels=1, filename='PC_label.pdf'):
    # plot in 3d using plotly
    df = pd.DataFrame(nodes_xyz, columns=['x', 'y', 'z'])
    # define custom colors for each category
    # colors = {'0': '#BCB6AE', '1': '#288596', '3': '#7D9083'}
    # colors = {'0': 'grey', '1': 'blue', '3': 'red'}
    # df['color'] = label.astype(int)
    # fig = px.scatter_3d(df, x='x', y='y', z='z', color = 'color', color_discrete_sequence=[colors[k] for k in sorted(colors.keys())])
    # # fig = px.scatter_3d(df, x='x', y='y', z='z', color = clr_nodes, color_continuous_scale=px.colors.sequential.Viridis)
    # fig.update_traces(marker_size = 1.5)  # increase marker_size for bigger node size
    # fig.show()   
    # plotly.offline.plot(fig)
    # fig.write_image(filename) 

    # Define custom colors for labels
    color_dict = {0: '#BCB6AE', 1: '#288596', 2: '#7D9083'}
    # color_dict = {0: '#BCB6AE', 1: '#288596'}
    colors = [color_dict[label] for label in labels]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['x'], df['y'], df['z'], c=colors, s=1.5)  
    ax.set_axis_off() # Hide coordinate space
    plt.savefig(filename)
    plt.close(fig)

def save_coord_for_visualization(data, savename):
    with open('./log/' + savename+'_LVendo.csv', 'w') as f:
        f.write('"Points:0","Points:1","Points:2"\n')
        for i in range(0, len(data)):
            f.write(str(data[i, 0]) + ',' + str(data[i, 1]) + ',' + str(data[i, 2]) + '\n')
    with open('./log/' + savename+'_epi.csv', 'w') as f:
        f.write('"Points:0","Points:1","Points:2"\n')
        for i in range(0, len(data)):
            f.write(str(data[i, 3]) + ',' + str(data[i, 4]) + ',' + str(data[i, 5]) + '\n')
    with open('./log/' + savename+'_RVendo.csv', 'w') as f:
        f.write('"Points:0","Points:1","Points:2"\n')
        for i in range(0, len(data)):
            f.write(str(data[i, 6]) + ',' + str(data[i, 7]) + ',' + str(data[i, 8]) + '\n')

def lossplot_detailed(lossfile_train, lossfile_val, lossfile_mesh_train, lossfile_mesh_val, lossfile_KL_train, lossfile_KL_val, lossfile_compactness_train, lossfile_compactness_val, lossfile_PC_train, lossfile_PC_val, lossfile_ecg_train, lossfile_ecg_val, lossfile_RVp_train, lossfile_RVp_val, lossfile_size_train, lossfile_size_val):
    ax = plt.subplot(331)
    ax.set_title('total loss')
    lossplot(lossfile_train, lossfile_val)

    ax = plt.subplot(332)
    ax.set_title('MI Dice + CE loss')
    lossplot(lossfile_mesh_train, lossfile_mesh_val)

    ax = plt.subplot(333)
    ax.set_title('MI compactness loss')
    lossplot(lossfile_compactness_train, lossfile_compactness_val)

    ax = plt.subplot(334)
    ax.set_title('KL loss')
    lossplot(lossfile_KL_train, lossfile_KL_val)

    ax = plt.subplot(335)
    ax.set_title('PC recon loss')
    lossplot(lossfile_PC_train, lossfile_PC_val)

    ax = plt.subplot(336)
    ax.set_title('ECG recon loss')
    lossplot(lossfile_ecg_train, lossfile_ecg_val)

    ax = plt.subplot(337)
    ax.set_title('MI size loss')
    lossplot(lossfile_size_train, lossfile_size_val)

    ax = plt.subplot(338)
    ax.set_title('MI RVpenalty loss')
    lossplot(lossfile_RVp_train, lossfile_RVp_val)

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

    plt.savefig("img.png")
    # plt.show()

def lossplot_classify(lossfile_train, lossfile_val, lossfile_mesh_train, lossfile_mesh_val, lossfile_KL_train, lossfile_KL_val, lossfile_ecg_train, lossfile_ecg_val):
    ax = plt.subplot(221)
    ax.set_title('total loss')
    lossplot(lossfile_train, lossfile_val)

    ax = plt.subplot(222)
    ax.set_title('MI classfication loss')
    lossplot(lossfile_mesh_train, lossfile_mesh_val)

    ax = plt.subplot(223)
    ax.set_title('KL loss')
    lossplot(lossfile_KL_train, lossfile_KL_val)


    ax = plt.subplot(224)
    ax.set_title('ECG recon loss')
    lossplot(lossfile_ecg_train, lossfile_ecg_val)


    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

    plt.savefig("img_classify.png")
    # plt.show()

def lossplot(lossfile1, lossfile2):
    loss = np.loadtxt(lossfile1)
    x = range(0, loss.size)
    y = loss
    plt.plot(x, y, '#FF7F61') # , label='train'
    plt.legend(frameon=False)

    loss = np.loadtxt(lossfile2)
    x = range(0, loss.size)
    y = loss
    plt.plot(x, y, '#2C4068') # , label='val'
    plt.legend(frameon=False)
    # plt.show()
    # plt.savefig("img.png")

def ECG_visual_two(prop_data, target_ecg):   
    prop_data[target_ecg[np.newaxis, ...] == 0.0], target_ecg[target_ecg == 0.0] = np.nan, np.nan

    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig, axs = plt.subplots(2, 8, constrained_layout=True, figsize=(40, 10))
    for i in range(8):
        leadName = leadNames[i]
        axs[0, i].plot(prop_data[0, i, :], color=[223/256,176/256,160/256], label='pred', linewidth=4)
        for j in range(1, prop_data.shape[0]):
            axs[0, i].plot(prop_data[j, i, :], color=[223/256,176/256,160/256], linewidth=4) 
        axs[0, i].plot(target_ecg[i, :], color=[154/256,181/256,174/256], label='true', linewidth=4)
        axs[0, i].set_title('Lead ' + leadName, fontsize=20)
        axs[0, i].set_axis_off() 
        axs[1, i].set_axis_off() 
    axs[0, i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    fig.savefig("ECG_visual.pdf")
    # plt.show()
    plt.close(fig)

if __name__ == '__main__':
    # input_data_dir = 'C:/Users/lilei/OneDrive - Nexus365/2021_Oxford/Oxford Research/BivenMesh_Script/dataset/gt/'
    # pc = input_data_dir + 'dense_RV_endo_output_labeled_ES_pc_6003744.ply'
    # pc_volume = calculate_pointcloudvolume(pc)
    # F_visual_CV()

    log_dir = 'E:/2022_ECG_inference/Cardiac_Personalisation/log'
    lossfile_train = log_dir + "/training_loss.txt"
    lossfile_val = log_dir + "/val_loss.txt"
    lossfile_geometry_train = log_dir + "/training_calculate_inference_loss.txt"
    lossfile_geometry_val = log_dir + "/val_calculate_inference_loss.txt"
    lossfile_compactness_train = log_dir + "/training_compactness_loss.txt"
    lossfile_compactness_val = log_dir + "/val_compactness_loss.txt"
    lossfile_KL_train = log_dir + "/training_KL_loss.txt"
    lossfile_KL_val = log_dir + "/val_KL_loss.txt"
    lossfile_PC_train = log_dir + "/training_PC_loss.txt"
    lossfile_PC_val = log_dir + "/val_PC_loss.txt"
    lossfile_ecg_train = log_dir + "/training_ecg_loss.txt"
    lossfile_ecg_val = log_dir + "/val_ecg_loss.txt"
    lossfile_RVp_train = log_dir + "/training_RVp_loss.txt"
    lossfile_RVp_val = log_dir + "/val_RVp_loss.txt"
    lossfile_size_train = log_dir + "/training_MIsize_loss.txt"
    lossfile_size_val = log_dir + "/val_MIsize_loss.txt"

    lossplot_detailed(lossfile_train, lossfile_val, lossfile_geometry_train, lossfile_geometry_val, lossfile_KL_train, lossfile_KL_val, lossfile_compactness_train, lossfile_compactness_val, lossfile_PC_train, lossfile_PC_val, lossfile_ecg_train, lossfile_ecg_val, lossfile_RVp_train, lossfile_RVp_val, lossfile_size_train, lossfile_size_val)
