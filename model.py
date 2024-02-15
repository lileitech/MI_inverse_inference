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

        # PointNet++ Encoder
        self.sa1 = PointNetSetAbstraction(npoint=num_input, radius=0.2, nsample=64, in_channel=in_ch, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 512, 1024], False)
        self.fc11 = nn.Linear(1024*16, z_dims*2)

        # PointNet++ Decoder
        self.fc12 = nn.Linear(z_dims*2, 1024) # feat_ECG = H*feat_MI + epsilon
        self.fp3 = PointNetFeaturePropagation(1280, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, out_ch, 1) 

        self.decoder_geometry = BetaVAE_Decoder(num_input, num_input//4, in_ch, z_dims) # in_ch -> out_ch*3
        
        self.encoder_signal = CRNN()

        # decode for signal
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(z_dims, 256*2)
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
        latent_z_signal = mu_signal # self.reparameterize(mu_signal, std_signal)

        # extract point cloud features      
        l0_xyz = partial_input[:,:3,:] 
        l0_points = partial_input[:,3:,:] 
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        features = self.fc11(l3_points.view(-1, 1024*16))
        mu_geometry = features[:, : self.z_dims]
        std_geometry = features[:, self.z_dims:] + 1e-6
        latent_geometry = self.reparameterize(mu_signal, std_signal)
        # latent_geometry = self.fc11(l3_points.view(-1, 1024*16))

        # fuse two features 
        # mu = torch.cat((mu_geometry, mu_signal), dim=1)
        # log_var = torch.cat((std_geometry, std_signal), dim=1)
        # latent_z = self.reparameterize(mu, log_var)
        latent_z = torch.cat((latent_z_signal, latent_geometry), dim=1)

        # segment point cloud
        anatomy_signal_feat = F.relu(self.fc12(latent_z))
        anatomy_signal_feat = anatomy_signal_feat.view(-1, 1024, 1).repeat(1, 1, num_points)      
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, anatomy_signal_feat)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        y_seg = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        y_seg = self.conv2(y_seg)   
        y_seg = nn.Softmax(dim=1)(y_seg)     

        # reconstruct point cloud and ecg
        y_coarse, y_detail = self.decoder_geometry(latent_geometry)
        y_coarse, y_detail = nn.Sigmoid()(y_coarse), nn.Sigmoid()(y_detail)  
        y_ECG = self.decode_signal(latent_z_signal)

        return y_seg, y_coarse, y_detail, y_ECG, mu_signal, std_signal

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

class PointNet(nn.Module):
    def __init__(self, num_classes=10, n_signal=10, n_param=4, n_ECG=128):
        super(PointNet, self).__init__()
        self.k = num_classes
        self.n_signal = n_signal
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=4)
        self.conv1 = torch.nn.Conv1d(1024+64+n_ECG, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.ECG_model = CRNN()

        self.inference_model = nn.Sequential(
            nn.Linear(1024+n_ECG, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, self.n_signal*n_param),
            nn.Sigmoid()
            )


    def forward(self, x, signal):
        n_pts = x.size()[2]
        anatomy_signal_feature, global_feature, trans_feat = self.feat(x)
        ECG_feature = self.ECG_model(signal)
        ECG_feature_extend = ECG_feature.repeat(1, 1, n_pts)
        
        anatomy_signal_feat = torch.cat([anatomy_signal_feature, ECG_feature_extend], 1)
        y1 = F.relu(self.bn1(self.conv1(anatomy_signal_feat)))
        y1 = F.relu(self.bn2(self.conv2(y1)))
        y1 = F.relu(self.bn3(self.conv3(y1)))
        y1 = self.conv4(y1)
        y1 = y1.transpose(2,1).contiguous()
        out_ATM = y1 #nn.Sigmoid()(y1)

        return out_ATM

class PointNet_plusplus(nn.Module):
    def __init__(self, num_classes=10, n_signal=10, n_param=4, n_ECG=128):
        super(PointNet_plusplus, self).__init__()
        self.n_signal = n_signal
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel= 3 + 4, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 512, 1024], False)
        self.fp3 = PointNetFeaturePropagation(1280+n_ECG, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        self.ECG_model = CRNN()
        self.inference_model = nn.Sequential(
            nn.Linear(1024+n_ECG, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, self.n_signal*n_param),
            nn.Sigmoid())

    def forward(self, x, signal):
        l0_points = x
        l0_xyz = x[:,:3,:] 

        ECG_feature = self.ECG_model(signal)
 
        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        ECG_feature_extend = ECG_feature.repeat(1, 1, l3_points.size()[2])  
        anatomy_signal_feat = torch.cat([l3_points, ECG_feature_extend], 1)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, anatomy_signal_feat)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        y1 = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        y1 = self.conv2(y1)        
        out_ATM = y1 #nn.Sigmoid()(y1)
        out_ATM = out_ATM.permute(0, 2, 1)

        return out_ATM

class BetaVAE(nn.Module):
    def __init__(self, in_ch=4, num_input=1024, num_class=2, z_dims=16):
        super(BetaVAE, self).__init__()

        self.encoder = BetaVAE_Encoder(in_ch, z_dims)
        self.decoder = BetaVAE_Decoder_new(num_input, num_class)

    def forward(self, x):
        latent_z = self.encoder(x)
        y = self.decoder(latent_z)
        return y

class BetaVAE_Encoder(nn.Module):
    def __init__(self, in_ch, z_dims):
        super(BetaVAE_Encoder, self).__init__()
        self.z_dims = z_dims
        self.mlp_conv1 = mlp_conv(in_ch, layer_dims=[128, 256])
        self.mlp_conv2 = mlp_conv(512, layer_dims=[512, 1024])

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, z_dims*2) 

    def forward(self, inputs):
        num_points = [inputs.shape[2]]
        features = self.mlp_conv1(inputs)
        features_global = point_maxpool(features, num_points, keepdim=True)
        features_global = point_unpool(features_global, num_points)
        features = torch.cat([features, features_global], dim=1)
        features = self.mlp_conv2(features)
        features = point_maxpool(features, num_points)

        features = features.view(features.size()[0], -1)
        features = self.fc1(features)
        features = self.fc2(features)
        features = self.fc3(features)
        mean = features[:, : self.z_dims]
        std = features[:, self.z_dims:] + 1e-6

        return mean, std

class BetaVAE_Decoder_new(nn.Module):
        def __init__(self, num_input, num_class=2, z_dims=16*2):
            super(BetaVAE_Decoder_new, self).__init__()
            self.out_ch = num_class
            self.n_pts = num_input 
            self.mlp = mlp(in_channels=z_dims, layer_dims=[128, 256, 512, 1024,  self.n_pts * self.out_ch]) 

        def forward(self, features):
            y = self.mlp(features).reshape(-1, self.out_ch, self.n_pts)
            
            return nn.Softmax(dim=1)(y)

class BetaVAE_Decoder_plus(nn.Module):
        def __init__(self, num_dense, num_coarse, out_ch, z_dims):
            super(BetaVAE_Decoder_plus, self).__init__()
            self.out_ch = out_ch
            self.num_coarse = num_coarse
            self.grid_size = int(np.sqrt(num_dense//num_coarse))
            self.num_fine = num_dense

            # PointNet++ Decoder
            self.fc12 = nn.Linear(z_dims*2, 1024)
            self.fp3 = PointNetFeaturePropagation(1280, [256, 256])
            self.fp2 = PointNetFeaturePropagation(384, [256, 128])
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
            self.conv1 = nn.Conv1d(128, 128, 1)
            self.bn1 = nn.BatchNorm1d(128)
            self.drop1 = nn.Dropout(0.5)
            self.conv2 = nn.Conv1d(128, out_ch, 1) 


        def forward(self, latent_z, l0_xyz, l1_xyz, l2_xyz, l3_xyz):
            anatomy_signal_feat = F.relu(self.fc12(latent_z))
            coarse = anatomy_signal_feat.view(-1, 1024, 1).repeat(1, 1, self.num_coarse)      
            l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, coarse)
            l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
            l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
            fine = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
            fine = self.conv2(fine)   

            return coarse, fine

class BetaVAE_Decoder(nn.Module):
        def __init__(self, num_dense, num_coarse, out_ch, z_dims):
            super(BetaVAE_Decoder, self).__init__()
            self.out_ch = out_ch
            self.num_coarse = num_coarse
            self.grid_size = int(np.sqrt(num_dense//num_coarse))
            self.num_fine = num_dense

            self.mlp = mlp(in_channels=z_dims, layer_dims=[256, 512, 1024, 2048,  self.num_coarse * self.out_ch])
            x = torch.linspace(-0.05, 0.05, self.grid_size)
            y = torch.linspace(-0.05, 0.05, self.grid_size)
            self.grid = torch.cat(torch.meshgrid(x, y), dim=0).view(1, 2, self.grid_size ** 2)
            # self.grid = torch.stack(torch.meshgrid(x, y), dim=2)
            # self.grid = torch.reshape(self.grid.transpose(1, 0), [-1, 2]).unsqueeze(0)

            self.mlp_conv3 = mlp_conv(z_dims+2+out_ch, layer_dims=[512, 512, out_ch]) # here "+2" refers  to the two axes of grid

        def forward(self, latent_z):
            features = latent_z
            coarse = self.mlp(features).reshape(-1, self.num_coarse, self.out_ch)
            point_feat = coarse.unsqueeze(2).repeat(1, 1, self.grid_size * 2, 1)
            point_feat = point_feat.reshape(-1, self.out_ch, self.num_fine)

            grid_feat = self.grid.unsqueeze(2).repeat(features.shape[0], 1, self.num_coarse, 1).to(features.device)
            grid_feat = grid_feat.reshape(features.shape[0], -1, self.num_fine) 
            global_feat = features.unsqueeze(2).repeat(1, 1, self.num_fine)
            feat = torch.cat([grid_feat, point_feat, global_feat], dim=1)
          
            center = point_feat.reshape(-1, self.num_fine, self.out_ch)
            fine = self.mlp_conv3(feat).transpose(1, 2) + center

            return coarse, fine

def point_maxpool(features, npts, keepdim=True):
    splitted = torch.split(features, npts[0], dim=1)
    outputs = [torch.max(f, dim=2, keepdim=keepdim)[0] for f in splitted] # modified by Lei in 2022/02/10
    return torch.cat(outputs, dim=0)
    # return torch.max(features, dim=2, keepdims=keepdims)[0]

def point_unpool(features, npts):
    features = torch.split(features, features.shape[0], dim=0)
    outputs = [f.repeat(1, 1, npts[i]) for i, f in enumerate(features)]
    # outputs = [torch.tile(f, [1, 1, npts[i]]) for i, f in enumerate(features)]
    return torch.cat(outputs, dim=0)
    # return features.repeat([1, 1, 256])

class mlp_conv(nn.Module):
    def __init__(self, in_channels, layer_dims):
        super(mlp_conv, self).__init__()
        self.layer_dims = layer_dims
        for i, out_channels in enumerate(self.layer_dims):
            layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
            setattr(self, 'conv_' + str(i), layer)
            in_channels = out_channels

    def __call__(self, inputs):
        outputs = inputs
        dims = len(self.layer_dims)
        for i in range(dims):
            layer = getattr(self, 'conv_' + str(i))
            if i == dims - 1:
                outputs = layer(outputs)
            else:
                outputs = relu(layer(outputs))
        return outputs

class mlp(nn.Module):
    def __init__(self, in_channels, layer_dims):
        super(mlp, self).__init__()
        self.layer_dims = layer_dims
        for i, out_channels in enumerate(layer_dims):
            layer = torch.nn.Linear(in_channels, out_channels)
            setattr(self, 'fc_' + str(i), layer)
            in_channels = out_channels

    def __call__(self, inputs):
        outputs = inputs
        dims = len(self.layer_dims)
        for i in range(dims):
            layer = getattr(self, 'fc_' + str(i))
            if i == dims - 1:
                outputs = layer(outputs)
            else:
                outputs = relu(layer(outputs))
        return outputs

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

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ELU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


if __name__ == "__main__":
    x = torch.rand(3, 4, 2048)
    conditions = torch.rand(3, 2, 1)

    network = BetaVAE()
    y_coarse, y_detail = network(x, conditions)
    print(y_coarse.size(), y_detail.size())
