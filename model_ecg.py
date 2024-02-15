from tkinter import Y
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu

from models.pointnet_utils import PointNetEncoder
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation

class ECGnet(nn.Module):
    def __init__(self, in_ch=3+4, out_ch=3, num_input=1024, z_dims=16):
        super(ECGnet, self).__init__()


        self.encoder_signal = CRNN()

        # decode for signal
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(z_dims, 256*2)
        self.fc2 = nn.Linear(256*2, 512*2)
        self.up = nn.Upsample(size=(8, 512), mode='bilinear')
        self.deconv = DoubleDeConv(1, 1)

        self.decoder_MI = nn.Sequential(
            nn.Linear(z_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_ch),
        )

        
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn(log_var.shape).to(std.device) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def decode_signal(self, latent_z): # P(x|z, c)
        '''
        z: (bs, latent_size)
        '''
        inputs = latent_z
        f = self.elu(self.fc1(inputs))
        f = self.elu(self.fc2(f))
        u = self.up(f.reshape(f.shape[0], 1, 8, -1))
        dc = self.deconv(u)

        return dc
    
    def forward(self, partial_input, signal_input):   

        mu_signal, std_signal = self.encoder_signal(signal_input)
        latent_z_signal = self.reparameterize(mu_signal, std_signal)
        y_ECG = self.decode_signal(latent_z_signal)
        y_MI = self.decoder_MI(latent_z_signal)
        y_MI = nn.Softmax(dim=1)(y_MI)

        return y_MI, y_ECG, mu_signal, std_signal

class InferenceNet(nn.Module):
    def __init__(self, in_ch=3+4, out_ch=3, num_input=1024, z_dims=16):
        super(InferenceNet, self).__init__()

        self.z_dims = z_dims

        # encode for signal
        self.encoder_signal = CRNN()

        # decode for signal
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(z_dims*2, 256*2)
        self.fc2 = nn.Linear(256*2, 512*2)
        self.up = nn.Upsample(size=(8, 512), mode='bilinear')
        self.deconv = DoubleDeConv(1, 1)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn(log_var.shape).to(std.device) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def decode_signal(self, latent_z): # P(x|z, c)
        '''
        z: (bs, latent_size)
        '''
        inputs = latent_z
        f = self.elu(self.fc1(inputs))
        f = self.elu(self.fc2(f))
        u = self.up(f.reshape(f.shape[0], 1, 8, -1))
        dc = self.deconv(u)

        return dc
    
    def forward(self, partial_input, signal_input):  
        num_points = partial_input.shape[-1]
        # extract ecg features
        mu_signal, std_signal = self.encoder_signal(signal_input)
        # latent_z_signal = self.reparameterize(mu_signal, std_signal)


        # fuse two features 
        mu = torch.cat((mu_geometry, mu_signal), dim=1)
        log_var = torch.cat((std_geometry, std_signal), dim=1)
        latent_z = self.reparameterize(mu, log_var)


        y_ECG = self.decode_signal(latent_z)

        return y_seg, y_coarse, y_detail, y_ECG, mu, log_var

class CRNN(nn.Module):
    '''
    nh: default=256, 'size of the LSTM hidden state'
    imgH: default=8, 'the height of the input image to network'
    imgW: default=256, 'the width of the input image to network'

    :param class_labels: list[n_class]
    :return: (n_batch, n_class)
    '''

    def __init__(self, n_lead=8, z_dims=16):
        super(CRNN, self).__init__()

        n_out = 128
        self.z_dims = z_dims

        self.cnn = nn.Sequential(
            nn.Conv1d(n_lead, n_out, kernel_size=16, stride=2, padding=2),
            nn.BatchNorm1d(n_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(n_out, n_out*2, kernel_size=16, stride=2, padding=2),
            nn.BatchNorm1d(n_out*2),
            nn.LeakyReLU(0.2, inplace=True)
            )
            

        self.rnn = BidirectionalLSTM(256, z_dims*4, z_dims*2)
        # self.rnn = nn.Sequential(
        #     BidirectionalLSTM(512, nh, nh),
        #     BidirectionalLSTM(nh, nh, 1))


    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, w = conv.size()
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv).permute(1, 0, 2)
        features = torch.max(output, 1)[0]
        mean = features[:, : self.z_dims]
        std = features[:, self.z_dims:] + 1e-6
    
        return mean, std


    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class DoubleDeConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleDeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

def dtw_loss(ecg1, ecg2): # to do: plot the curve of x-y axis.
    """
    计算两个ECG序列之间的Dynamic Time Warping（DTW）损失。

    参数：
    - ecg1: 第一个ECG序列，形状为 (batch_size, seq_len1, num_features)
    - ecg2: 第二个ECG序列，形状为 (batch_size, seq_len2, num_features)

    返回：
    - dtw_loss: DTW损失，标量张量
    """
    batch_size, seq_len1, num_features = ecg1.size()
    _, seq_len2, _ = ecg2.size()

    # 计算两个ECG序列之间的距离矩阵
    distance_matrix = torch.cdist(ecg1, ecg2)  # 形状为 (batch_size, seq_len1, seq_len2)

    # 初始化动态规划表格
    torch.autograd.set_detect_anomaly(True)
    dp = torch.zeros((batch_size, seq_len1, seq_len2)).to(ecg1.device)

    # 填充动态规划表格
    dp[:, 0, 0] = distance_matrix[:, 0, 0]
    for i in range(1, seq_len1):
        dp[:, i, 0] = distance_matrix[:, i, 0] + dp[:, i-1, 0].clone()
    for j in range(1, seq_len2):
        dp[:, 0, j] = distance_matrix[:, 0, j] + dp[:, 0, j-1].clone()
    for i in range(1, seq_len1):
        for j in range(1, seq_len2):
            dp[:, i, j] = distance_matrix[:, i, j] + torch.min(torch.stack([
                dp[:, i-1, j].clone(),
                dp[:, i, j-1].clone(),
                dp[:, i-1, j-1].clone()
            ], dim=1), dim=1).values

    dtw_loss = torch.mean(dp[:, seq_len1-1, seq_len2-1] / (seq_len1 + seq_len2))

    return dtw_loss

if __name__ == "__main__":
    x = torch.rand(3, 4, 2048)
    conditions = torch.rand(3, 2, 1)

    network = BetaVAE()
    y_coarse, y_detail = network(x, conditions)
    print(y_coarse.size(), y_detail.size())
