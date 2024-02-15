import argparse
import torch
torch.cuda.empty_cache() # clearing the occupied cuda memory
from torch.backends import cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"


from dataset import LoadDataset
from model import InferenceNet, ECGnet
from loss import calculate_inference_loss, calculate_reconstruction_loss, calculate_ECG_reconstruction_loss, calculate_classify_loss
from utils import lossplot, lossplot_detailed, visualize_PC_with_label, ECG_visual_two, lossplot_classify, visualize_PC_with_twolabel

def train_ecg(args):
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # DEVICE = torch.device('cpu')
    train_dataset = LoadDataset(path=args.partial_root, num_input=args.num_input, split='train')
    val_dataset = LoadDataset(path=args.partial_root, num_input=args.num_input, split='val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    cudnn.benchmark = True

    network = InferenceNet(in_ch=args.in_ch, out_ch=args.out_ch, num_input=args.num_input, z_dims=args.z_dims)

    if args.model is not None:
        print('Loaded trained model from {}.'.format(args.model))
        network.load_state_dict(torch.load(args.model))
    else:
        print('Begin training new model.')

    network.to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)

    max_iter = int(len(train_dataset) / args.batch_size + 0.5)
    minimum_loss = 1e4
    best_epoch = 0

    lossfile_train = args.log_dir + "/training_loss.txt"
    lossfile_val = args.log_dir + "/val_loss.txt"
    lossfile_geometry_train = args.log_dir + "/training_calculate_inference_loss.txt"
    lossfile_geometry_val = args.log_dir + "/val_calculate_inference_loss.txt"
    lossfile_KL_train = args.log_dir + "/training_KL_loss.txt"
    lossfile_KL_val = args.log_dir + "/val_KL_loss.txt"
    lossfile_ecg_train = args.log_dir + "/training_ecg_loss.txt"
    lossfile_ecg_val = args.log_dir + "/val_ecg_loss.txt"


    for epoch in range(1, args.epochs + 1):
        if ((epoch % 25) == 0) and (epoch != 0):  
            lossplot_classify(lossfile_train, lossfile_val, lossfile_geometry_train, lossfile_geometry_val, lossfile_KL_train, lossfile_KL_val, lossfile_ecg_train, lossfile_ecg_val)

        f_train = open(lossfile_train, 'a')  # a: additional writing; w: overwrite writing
        f_val = open(lossfile_val, 'a')
        f_MI_train = open(lossfile_geometry_train, 'a')  # a: additional writing; w: overwrite writing
        f_MI_val = open(lossfile_geometry_val, 'a')
        f_KL_train = open(lossfile_KL_train, 'a')  # a: additional writing; w: overwrite writing
        f_KL_val = open(lossfile_KL_val, 'a')
        f_ecg_train = open(lossfile_ecg_train, 'a')  # a: additional writing; w: overwrite writing
        f_ecg_val = open(lossfile_ecg_val, 'a')

        # if ((epoch % 25) == 0) and (epoch != 0): 
        #     if  lamda_KL < 1:
        #         lamda_KL = 0.1*epoch*lamda_KL # 0.25
        #     else:
        #         lamda_KL = 0.1

        # training
        network.train()
        total_loss, iter_count = 0, 0
        for i, data in enumerate(train_dataloader, 1):
            partial_input, ECG_input, gt_MI, partial_input_coarse = data
            partial_input, ECG_input, gt_MI = partial_input.to(DEVICE), ECG_input.to(DEVICE), gt_MI.to(DEVICE)      
            partial_input_coarse = partial_input_coarse.to(DEVICE)      
            partial_input = partial_input.permute(0, 2, 1)

            optimizer.zero_grad()

            y_MI, y_ECG, mu, log_var = network(partial_input, ECG_input)
       
            loss_seg, KL_loss = calculate_classify_loss(y_MI, gt_MI, mu, log_var)
            loss_signal = calculate_ECG_reconstruction_loss(y_ECG, ECG_input)
            loss = loss_seg + args.lamda_KL*KL_loss

            check_grad = False
            if check_grad:
                print(loss_seg)
                print(loss_signal)
                print(KL_loss)

                print(loss.requires_grad)
                print(loss_seg.requires_grad)
                print(KL_loss.requires_grad)
                print(loss_signal.requires_grad)

            visual_check = False
            if visual_check:
                gd_ECG = ECG_input[0].cpu().detach().numpy()
                y_ECG = y_ECG[0].cpu().detach().numpy()
                ECG_visual_two(y_ECG, gd_ECG)
                
            loss.backward()
            optimizer.step()

            f_train.write(str(loss.item()))
            f_train.write('\n')
            f_MI_train.write(str(loss_seg.item()))
            f_MI_train.write('\n')
            f_KL_train.write(str(KL_loss.item()))
            f_KL_train.write('\n')
            f_ecg_train.write(str(loss_signal.item()))
            f_ecg_train.write('\n')


            iter_count += 1
            total_loss += loss.item()

            if i % 50 == 0:
                print("Training epoch {}/{}, iteration {}/{}: loss is {}".format(epoch, args.epochs, i, max_iter, loss.item()))
        scheduler.step()

        print("\033[96mTraining epoch {}/{}: avg loss = {}\033[0m".format(epoch, args.epochs, total_loss / iter_count))

        # evaluation
        network.eval()
        with torch.no_grad():
            total_loss, iter_count = 0, 0
            for i, data in enumerate(val_dataloader, 1):
                partial_input, ECG_input, gt_MI, partial_input_coarse = data
                partial_input, ECG_input, gt_MI = partial_input.to(DEVICE), ECG_input.to(DEVICE), gt_MI.to(DEVICE)  
                partial_input_coarse = partial_input_coarse.to(DEVICE)  
                partial_input = partial_input.permute(0, 2, 1)

                y_MI, y_ECG, mu, log_var = network(partial_input, ECG_input)
        
                loss_seg, KL_loss = calculate_classify_loss(y_MI, gt_MI, mu, log_var)
                loss_signal = calculate_ECG_reconstruction_loss(y_ECG, ECG_input)
                loss = loss_seg + args.lamda_KL*KL_loss

                total_loss += loss.item()
                iter_count += 1

                visual_check = False
                if visual_check:
                    gd_ECG = ECG_input[0].cpu().detach().numpy()
                    y_ECG = y_ECG[0].cpu().detach().numpy()
                    ECG_visual_two(y_ECG, gd_ECG)
                    
                f_val.write(str(loss.item()))
                f_val.write('\n')
                f_MI_val.write(str(loss_seg.item()))
                f_MI_val.write('\n')
                f_KL_val.write(str(KL_loss.item()))
                f_KL_val.write('\n')
                f_ecg_val.write(str(loss_signal.item()))
                f_ecg_val.write('\n')
    

            mean_loss = total_loss / iter_count
            print("\033[35mValidation epoch {}/{}, loss is {}\033[0m".format(epoch, args.epochs, mean_loss))

            # records the best model and epoch
            if mean_loss < minimum_loss:
                best_epoch = epoch
                minimum_loss = mean_loss           
                strNetSaveName = 'net_model_classify.pkl'
                # strNetSaveName = 'net_with_%d.pkl' % epoch
                torch.save(network.state_dict(), args.log_dir + '/' + strNetSaveName)

        print("\033[4;37mBest model (lowest loss) in epoch {}\033[0m".format(best_epoch))

    lossplot(lossfile_train, lossfile_val)


def train(args):
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # DEVICE = torch.device('cpu')
    train_dataset = LoadDataset(path=args.partial_root, num_input=args.num_input, split='train')
    val_dataset = LoadDataset(path=args.partial_root, num_input=args.num_input, split='val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    cudnn.benchmark = True

    network = InferenceNet(in_ch=args.in_ch, out_ch=args.out_ch, num_input=args.num_input, z_dims=args.z_dims)

    if args.model is not None:
        print('Loaded trained model from {}.'.format(args.model))
        network.load_state_dict(torch.load(args.model))
    else:
        print('Begin training new model.')

    network.to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)

    max_iter = int(len(train_dataset) / args.batch_size + 0.5)
    minimum_loss = 1e4
    best_epoch = 0

    lossfile_train = args.log_dir + "/training_loss.txt"
    lossfile_val = args.log_dir + "/val_loss.txt"
    lossfile_geometry_train = args.log_dir + "/training_calculate_inference_loss.txt"
    lossfile_geometry_val = args.log_dir + "/val_calculate_inference_loss.txt"
    lossfile_compactness_train = args.log_dir + "/training_compactness_loss.txt"
    lossfile_compactness_val = args.log_dir + "/val_compactness_loss.txt"
    lossfile_KL_train = args.log_dir + "/training_KL_loss.txt"
    lossfile_KL_val = args.log_dir + "/val_KL_loss.txt"
    lossfile_PC_train = args.log_dir + "/training_PC_loss.txt"
    lossfile_PC_val = args.log_dir + "/val_PC_loss.txt"
    lossfile_ecg_train = args.log_dir + "/training_ecg_loss.txt"
    lossfile_ecg_val = args.log_dir + "/val_ecg_loss.txt"
    lossfile_RVp_train = args.log_dir + "/training_RVp_loss.txt"
    lossfile_RVp_val = args.log_dir + "/val_RVp_loss.txt"
    lossfile_size_train = args.log_dir + "/training_MIsize_loss.txt"
    lossfile_size_val = args.log_dir + "/val_MIsize_loss.txt"

    lamda_KL = args.lamda_KL
    for epoch in range(1, args.epochs + 1):
        if ((epoch % 25) == 0) and (epoch != 0):  
            lossplot_detailed(lossfile_train, lossfile_val, lossfile_geometry_train, lossfile_geometry_val, lossfile_KL_train, lossfile_KL_val, lossfile_compactness_train, lossfile_compactness_val, lossfile_PC_train, lossfile_PC_val, lossfile_ecg_train, lossfile_ecg_val, lossfile_RVp_train, lossfile_RVp_val, lossfile_size_train, lossfile_size_val)

        f_train = open(lossfile_train, 'a')  # a: additional writing; w: overwrite writing
        f_val = open(lossfile_val, 'a')
        f_MI_train = open(lossfile_geometry_train, 'a')  # a: additional writing; w: overwrite writing
        f_MI_val = open(lossfile_geometry_val, 'a')
        f_compactness_train = open(lossfile_compactness_train, 'a')  # a: additional writing; w: overwrite writing
        f_compactness_val = open(lossfile_compactness_val, 'a')
        f_KL_train = open(lossfile_KL_train, 'a')  # a: additional writing; w: overwrite writing
        f_KL_val = open(lossfile_KL_val, 'a')
        f_PC_train = open(lossfile_PC_train, 'a')  # a: additional writing; w: overwrite writing
        f_PC_val = open(lossfile_PC_val, 'a')
        f_ecg_train = open(lossfile_ecg_train, 'a')  # a: additional writing; w: overwrite writing
        f_ecg_val = open(lossfile_ecg_val, 'a')
        f_size_train = open(lossfile_size_train, 'a')  # a: additional writing; w: overwrite writing
        f_size_val = open(lossfile_size_val, 'a')
        f_RVp_train = open(lossfile_RVp_train, 'a')  # a: additional writing; w: overwrite writing
        f_RVp_val = open(lossfile_RVp_val, 'a')

        # if epoch != 0: 
        #     if  lamda_KL < 1:
        #         lamda_KL = 0.1*epoch*args.lamda_KL 
        #     else:
        #         lamda_KL = 0.1

        # training
        network.train()
        total_loss, iter_count = 0, 0
        for i, data in enumerate(train_dataloader, 1):
            partial_input, ECG_input, gt_MI, partial_input_coarse, MI_type = data
            partial_input, ECG_input, gt_MI = partial_input.to(DEVICE), ECG_input.to(DEVICE), gt_MI.to(DEVICE)      
            partial_input_coarse = partial_input_coarse.to(DEVICE)      
            partial_input = partial_input.permute(0, 2, 1)

            optimizer.zero_grad()

            y_MI, y_coarse, y_detail, y_ECG, mu, log_var = network(partial_input[:, 0:7, :], ECG_input)
       
            loss_seg, loss_compactness, loss_MI_RVpenalty, loss_MI_size, KL_loss = calculate_inference_loss(y_MI, gt_MI, mu, log_var, partial_input)
            loss_geo, loss_signal = calculate_reconstruction_loss(y_coarse, y_detail, partial_input_coarse, partial_input, y_ECG, ECG_input)
            loss = loss_seg + args.lamda_compact*loss_compactness + args.lamda_RVp*loss_MI_RVpenalty + args.lamda_MIsize*loss_MI_size + args.lamda_KL*KL_loss + args.lamda_recon*loss_geo # + args.lamda_recon*loss_signal # 

            check_grad = False
            if check_grad:
                print(loss.requires_grad)
                print(loss_seg.requires_grad)
                print(loss_compactness.requires_grad)
                print(loss_MI_RVpenalty.requires_grad)
                print(KL_loss.requires_grad)
                print(loss_MI_size.requires_grad)
                print(loss_geo.requires_grad)
                print(loss_signal.requires_grad)

            visual_check = False
            if visual_check:
                y_predict = y_MI[0].cpu().detach().numpy()
                y_gd = gt_MI[0].cpu().detach().numpy()
                x_input = partial_input[0].cpu().detach().numpy()
                y_predict_argmax = np.argmax(y_predict, axis=0)
                visualize_PC_with_twolabel(x_input[0:3, 0:args.num_input].transpose(), y_predict_argmax, y_gd, filename='RNmap_gd_pre.jpg')

            loss.backward()
            optimizer.step()

            f_train.write(str(loss.item()))
            f_train.write('\n')
            f_MI_train.write(str(loss_seg.item()))
            f_MI_train.write('\n')
            f_compactness_train.write(str(loss_compactness.item()))
            f_compactness_train.write('\n')
            f_KL_train.write(str(KL_loss.item()))
            f_KL_train.write('\n')
            f_PC_train.write(str(loss_geo.item()))
            f_PC_train.write('\n')
            f_ecg_train.write(str(loss_signal.item()))
            f_ecg_train.write('\n')
            f_size_train.write(str((loss_MI_size.item())))
            f_size_train.write('\n')
            f_RVp_train.write(str(loss_MI_RVpenalty.item()))
            f_RVp_train.write('\n')

            iter_count += 1
            total_loss += loss.item()

            if i % 50 == 0:
                print("Training epoch {}/{}, iteration {}/{}: loss is {}".format(epoch, args.epochs, i, max_iter, loss.item()))
        scheduler.step()

        print("\033[96mTraining epoch {}/{}: avg loss = {}\033[0m".format(epoch, args.epochs, total_loss / iter_count))

        # evaluation
        network.eval()
        with torch.no_grad():
            total_loss, iter_count = 0, 0
            for i, data in enumerate(val_dataloader, 1):
                partial_input, ECG_input, gt_MI, partial_input_coarse, MI_type = data
                partial_input, ECG_input, gt_MI = partial_input.to(DEVICE), ECG_input.to(DEVICE), gt_MI.to(DEVICE)  
                partial_input_coarse = partial_input_coarse.to(DEVICE)  
                partial_input = partial_input.permute(0, 2, 1)

                y_MI, y_coarse, y_detail, y_ECG, mu, log_var = network(partial_input[:, 0:7, :], ECG_input)

                loss_seg, loss_compactness, loss_MI_RVpenalty, loss_MI_size, KL_loss = calculate_inference_loss(y_MI, gt_MI, mu, log_var, partial_input)
                loss_geo, loss_signal = calculate_reconstruction_loss(y_coarse, y_detail, partial_input_coarse, partial_input, y_ECG, ECG_input)
                loss = loss_seg + args.lamda_compact*loss_compactness + args.lamda_RVp*loss_MI_RVpenalty + args.lamda_MIsize*loss_MI_size + args.lamda_KL*KL_loss + args.lamda_recon*loss_geo # + args.lamda_recon*loss_signal # 

                total_loss += loss.item()
                iter_count += 1

                if ((epoch % 25) == 0) and (epoch != 0) and (i == 1):  
                    y_predict = y_MI[0].cpu().detach().numpy()
                    y_gd = gt_MI[0].cpu().detach().numpy()
                    x_input = partial_input[0].cpu().detach().numpy()
                    y_predict_argmax = np.argmax(y_predict, axis=0)
                    visualize_PC_with_twolabel(x_input[0:3, 0:args.num_input].transpose(), y_predict_argmax, y_gd, filename='RNmap_gd_pre.jpg')
                    
                f_val.write(str(loss.item()))
                f_val.write('\n')
                f_MI_val.write(str(loss_seg.item()))
                f_MI_val.write('\n')
                f_compactness_val.write(str(loss_compactness.item()))
                f_compactness_val.write('\n')
                f_KL_val.write(str(KL_loss.item()))
                f_KL_val.write('\n')
                f_PC_val.write(str(loss_geo.item()))
                f_PC_val.write('\n')
                f_ecg_val.write(str(loss_signal.item()))
                f_ecg_val.write('\n')
                f_size_val.write(str(loss_MI_size.item()))
                f_size_val.write('\n')
                f_RVp_val.write(str(loss_MI_RVpenalty.item()))
                f_RVp_val.write('\n')

            mean_loss = total_loss / iter_count
            print("\033[35mValidation epoch {}/{}, loss is {}\033[0m".format(epoch, args.epochs, mean_loss))

            # records the best model and epoch
            if mean_loss < minimum_loss:
                best_epoch = epoch
                minimum_loss = mean_loss           
                strNetSaveName = 'net_model.pkl'
                # strNetSaveName = 'net_with_%d.pkl' % epoch
                torch.save(network.state_dict(), args.log_dir + '/' + strNetSaveName)

        print("\033[4;37mBest model (lowest loss) in epoch {}\033[0m".format(best_epoch))

    lossplot(lossfile_train, lossfile_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial_root', type=str, default='./Big_data_inference/meta_data/UKB_clinical_data/')
    parser.add_argument('--model', type=str, default=None) #'log/net_model.pkl'
    parser.add_argument('--in_ch', type=int, default=3+4) # coordinate dimension + label index
    parser.add_argument('--out_ch', type=int, default=3) # 3scar, BZ, normal/ 18 for ecg-based classification
    parser.add_argument('--z_dims', type=int, default=16)
    parser.add_argument('--num_input', type=int, default=1024*4)
    parser.add_argument('--batch_size', type=int, default=4) # 4
    parser.add_argument('--lamda_recon', type=float, default=1) # 1
    parser.add_argument('--lamda_KL', type=float, default=1e-2) # 1e-2
    parser.add_argument('--lamda_MIsize', type=float, default=1) # 1
    parser.add_argument('--lamda_RVp', type=float, default=1) # 1 
    parser.add_argument('--lamda_compact', type=float, default=1) # 1
    parser.add_argument('--base_lr', type=float, default=1e-4) #1e-4
    parser.add_argument('--lr_decay_steps', type=int, default=50) 
    parser.add_argument('--lr_decay_rate', type=float, default=0.5) 
    parser.add_argument('--weight_decay', type=float, default=1e-3) #1e-3
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='log')
    args = parser.parse_args()

    train(args)