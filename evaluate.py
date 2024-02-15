import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from dataset import LoadDataset
from model import InferenceNet, ECGnet
from utils import visualize_two_PC, ECG_visual_two, visualize_PC_with_twolabel_rotated
from loss import calculate_Dice, evaluate_pointcloud, calculate_inference_loss, calculate_reconstruction_loss

def evaluate(args):

    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    test_dataset = LoadDataset(path=args.partial_root, num_input=args.num_input, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # network = ECGnet(in_ch=args.in_ch, out_ch=args.out_ch, num_input=args.num_input, z_dims=args.z_dims)
    network = InferenceNet(in_ch=args.in_ch, out_ch=args.out_ch, num_input=args.num_input, z_dims=args.z_dims)

    network.load_state_dict(torch.load('log/net_model.pkl'))
    network.to(DEVICE)

    Dice_Scar, Dice_BZ =  [], []
    precision_Scar, precision_BZ = [], []
    recall_Scar, recall_BZ = [], []
    f1_score_Scar, f1_score_BZ = [], []
    roc_auc_Scar, roc_auc_BZ = [], []
    pre_MI_size_Scar, pre_MI_size_BZ = [], []
    gd_MI_size_Scar, gd_MI_size_BZ = [], []
    MI_center_dist = []
    MI_type_list = []
    AHA_loc_score_list = []
    recon_geo_list, recon_ECG_list = [], []

    # testing: evaluate the mean loss
    network.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 1):
            partial_input, ECG_input, gt_MI, partial_input_coarse, MI_type = data
            partial_input, ECG_input, gt_MI = partial_input.to(DEVICE), ECG_input.to(DEVICE), gt_MI.to(DEVICE)      
            partial_input_coarse = partial_input_coarse.to(DEVICE)      
            partial_input = partial_input.permute(0, 2, 1)

            y_MI, y_coarse, y_detail, y_ECG, mu, log_var = network(partial_input[:, 0:7, :], ECG_input)
            loss_geo, loss_signal = calculate_reconstruction_loss(y_coarse, y_detail, partial_input_coarse, partial_input, y_ECG, ECG_input)

            Dice = calculate_Dice(y_MI, gt_MI, num_classes=3)
            precision, recall, f1_score, roc_auc, MI_size_pre, MI_size_gd, center_distance, AHA_loc_score = evaluate_pointcloud(y_MI, gt_MI, partial_input)

            Dice_Scar.append(Dice[1].cpu().detach().numpy())
            Dice_BZ.append(Dice[2].cpu().detach().numpy())
            precision_Scar.append(precision[1])
            precision_BZ.append(precision[2])
            recall_Scar.append(recall[1])
            recall_BZ.append(recall[2])
            # f1_score_Scar.append(f1_score[1])
            # f1_score_BZ.append(f1_score[2])
            # roc_auc_Scar.append(roc_auc[1])
            # roc_auc_BZ.append(roc_auc[2])

            pre_MI_size_Scar.append(MI_size_pre[1])
            pre_MI_size_BZ.append(MI_size_pre[2])
            gd_MI_size_Scar.append(MI_size_gd[1])
            gd_MI_size_BZ.append(MI_size_gd[2])
            MI_center_dist.append(center_distance)
            AHA_loc_score_list.append(AHA_loc_score)
            recon_geo_list.append(loss_geo.cpu().detach().numpy())
            recon_ECG_list.append(loss_signal.cpu().detach().numpy())

            MI_type_list.append(MI_type[0])

            visual_check = False
            if visual_check:
                gd_ECG = ECG_input[0].cpu().detach().numpy()
                y_ECG = y_ECG[0].cpu().detach().numpy()
                ECG_visual_two(y_ECG, gd_ECG)
                y_predict = y_MI[0].cpu().detach().numpy()
                y_gd = gt_MI[0].cpu().detach().numpy()
                x_input = partial_input[0].cpu().detach().numpy()
                y_predict_argmax = np.argmax(y_predict, axis=0)
                y_output = y_detail.permute(0, 2, 1)[0].cpu().detach().numpy()
                visualize_PC_with_twolabel_rotated(x_input[0:3, 0:args.num_input].transpose(), y_predict_argmax, y_gd, filename='RNmap_gd_pre.pdf')
                visualize_two_PC(x_input[0:3, 0:args.num_input].transpose(), y_output[0:3, 0:args.num_input].transpose(), y_gd, filename='PC_recon.pdf')

        list = {'MI_type': MI_type_list, 'Dice_Scar': Dice_Scar, 'Dice_BZ': Dice_BZ, 'precision_Scar': precision_Scar, 'precision_BZ': precision_BZ, 
        'recall_Scar': recall_Scar, 'recall_BZ': recall_BZ, 
        'pre_MI_size_Scar': pre_MI_size_Scar, 'pre_MI_size_BZ': pre_MI_size_BZ,
        'gd_MI_size_Scar': gd_MI_size_Scar, 'gd_MI_size_BZ': gd_MI_size_BZ
        , 'MI_center_dist': MI_center_dist, 'AHA_loc_score': AHA_loc_score_list
        , 'recon_geo': recon_geo_list, 'recon_ECG': recon_ECG_list}
        
        df = pd.DataFrame(list)
        df.to_csv('MI_inference_results_sample4.csv', encoding='gbk', index=False)

        print('Lei, well done!')       
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial_root', type=str, default='E:/OneDrive - Nexus365/PeoPle/Julia_Camps/Big_data_inference/meta_data/UKB_clinical_data/')
    parser.add_argument('--model', type=str, default='log/net_model.pkl') #'log/net_model.pkl'
    parser.add_argument('--in_ch', type=int, default=3+4) # coordinate dimension + label index
    parser.add_argument('--out_ch', type=int, default=3) # scar, BZ, normal
    parser.add_argument('--z_dims', type=int, default=16)
    parser.add_argument('--num_input', type=int, default=1024*4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--lamda', type=float, default=1)
    parser.add_argument('--base_lr', type=float, default=1e-5) #5e-5
    parser.add_argument('--lr_decay_steps', type=int, default=50) 
    parser.add_argument('--lr_decay_rate', type=float, default=0.5) 
    parser.add_argument('--weight_decay', type=float, default=1e-6) #1e-3
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='log')
    args = parser.parse_args()

    evaluate(args)

