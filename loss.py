import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import visualize_PC_with_label
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from utils import visualize_PC_with_label
# from distance.chamfer_distance import ChamferDistanceFunction
# from distance.emd_module import emdFunction

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

def calculate_classify_loss(y_MI, gt_MI_label, mu, log_var):

    loss_func_CE = nn.CrossEntropyLoss() # weight=PC_weight
    loss_CE = loss_func_CE(y_MI, gt_MI_label)

    KL_loss = -0.5 * torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var))

    return loss_CE, KL_loss

def calculate_ECG_reconstruction_loss(y_signal, signal_input):

    y_signal = y_signal.squeeze(1)

    loss_signal = torch.mean(torch.square(y_signal-signal_input))
    
    return loss_signal

def calculate_reconstruction_loss(y_coarse, y_detail, coarse_gt, dense_gt, y_signal, signal_input):
    dense_gt = dense_gt.permute(0, 2, 1)
    y_signal = y_signal.squeeze(1)
    loss_coarse = calculate_chamfer_distance(y_coarse[:, :, 0:3], coarse_gt[:, :, 0:3]) + calculate_chamfer_distance(y_coarse[:, :, 3:], coarse_gt[:, :, 3:7])
    loss_fine = calculate_chamfer_distance(y_detail[:, :, 0:3], dense_gt[:, :, 0:3]) + calculate_chamfer_distance(y_coarse[:, :, 3:], coarse_gt[:, :, 3:7])
    # loss_coarse_emd = calculate_emd(y_coarse[:, :, 0:3], coarse_gt[:, :, 0:3]) + calculate_emd(y_coarse[:, :, 3:], coarse_gt[:, :, 3:])

    # Per-class chamfer losses as reconstruction loss
    # loss_coarse = per_class_PCdist(y_coarse, coarse_gt, dist_type='chamfer') + per_class_PCdist(y_coarse, coarse_gt, dist_type = 'EDM')
    # loss_fine = per_class_PCdist(y_detail, dense_gt, dist_type='chamfer')

    loss_signal = torch.mean(torch.square(y_signal-signal_input)) 
    loss_DTW = dtw_loss(y_signal, signal_input) # dynamic time warping

    # ECG_dist = torch.sqrt(torch.sum((y_signal - signal_input) ** 2)) 
    # PC_dist = torch.sqrt(torch.sum((y_coarse[:, :, 3:7] - coarse_gt[:, :, 3:7]) ** 2)) + torch.sqrt(torch.sum((y_detail[:, :, 3:7] - dense_gt[:, :, 3:7]) ** 2))

    return loss_coarse + 5*loss_fine, loss_signal + loss_DTW #0.5*(loss_coarse + loss_fine), loss_signal + loss_DTW # 

def evaluate_AHA_localization(predicted_center_id, predicted_covered_ids, gt_center_id, gt_covered_ids, center_distance):
    # Center ID Comparison
    center_id_match = predicted_center_id == gt_center_id
    center_id_score = 1 if center_id_match else 0

    # Covered ID Comparison
    common_ids = set(predicted_covered_ids.tolist()) & set(gt_covered_ids.tolist())
    intersection = len(common_ids)
    union = len(set(predicted_covered_ids.tolist()).union(set(gt_covered_ids.tolist())))
    iou_score = intersection / union if union != 0 else 0

    # Weighting
    center_id_weight = 0.5
    center_distance_weight = 0.3
    covered_id_weight = 0.2

    # Overall Evaluation Metric
    evaluation_metric = (center_id_weight * center_id_score) + (covered_id_weight * iou_score) + (center_distance_weight*(1-center_distance))

    return evaluation_metric

def evaluate_pointcloud(predictions, target, partial_input, n_classes=3):
    # To address the issue of class imbalance and obtain a more comprehensive evaluation of model performance, 
    # you may consider using other metrics such as precision, recall (or sensitivity), F1-score, and area under the
    # receiver operating characteristic (ROC) curve. These metrics provide a more nuanced evaluation of model performance, 
    # taking into account both true positive and false positive/negative rates for each class separately.

    PC_xyz = partial_input[:, 0:3, :].permute(0, 2, 1).squeeze(0)
    AHA_id = partial_input[:, 7, :].squeeze(0)
    
    targets = F.one_hot(target, n_classes).permute(0, 2, 1)

    """Function to evaluate point cloud predictions with multiple classes"""
    assert predictions.shape == targets.shape, "Input shapes must be the same"
    assert predictions.shape[0] == 1, "Batch size must be 1"

    # Convert predictions and targets to boolean values based on threshold
    # predictions = torch.ge(predictions, threshold).bool()
    predictions = one_hot_argmax(predictions).bool()
    targets = targets.bool().squeeze(0)

    MI_size_pre = torch.sum(predictions, dim=1).tolist()
    MI_size_gd = torch.sum(targets, dim=1).tolist()

    y_MI_center = torch.mean(PC_xyz[predictions[1]], dim=0)
    gt_MI_center = torch.mean(PC_xyz[targets[1]], dim=0)

    # calculate and compare the covered AHA IDs and the centered AHA ID of prediction and ground truth
    kdtree = KDTree(PC_xyz.cpu().detach().numpy())
    distance_pre, index_pre = kdtree.query(y_MI_center.cpu().detach().numpy())
    distance_gd, index_gd = kdtree.query(gt_MI_center.cpu().detach().numpy()) # to do: check whether its AHA=0
    max_distance = torch.max(torch.sqrt(torch.sum((PC_xyz[AHA_id!=0.0][:, None] - PC_xyz[AHA_id!=0.0]) ** 2, dim=2)))
    if index_pre == 4096:
        center_distance = 1
        AHA_center_pre = 0
        print('no valid nearest neighbor was found')
    else:
        center_distance = (torch.sqrt(torch.sum((PC_xyz[index_pre] - PC_xyz[index_gd]) ** 2))/max_distance).cpu().detach().numpy()
        AHA_center_pre = AHA_id[index_pre] 
    AHA_center_gd = AHA_id[index_gd]
    AHA_list_pre, AHA_list_gd = torch.unique(AHA_id[predictions[1]]), torch.unique(AHA_id[targets[1]])
    AHA_loc_score = evaluate_AHA_localization(AHA_center_pre, AHA_list_pre, AHA_center_gd, AHA_list_gd, center_distance)

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN) for each class
    tp = torch.sum(predictions & targets, dim=1).tolist()
    fp = torch.sum(predictions & ~targets, dim=1).tolist()
    fn = torch.sum(~predictions & targets, dim=1).tolist()
    tn = torch.sum(~predictions & ~targets, dim=1).tolist()

    # Calculate Accuracy, Precision, Recall (Sensitivity), Specificity, and F1-score for each class
    accuracy = sum(tp) / (sum(tp) + sum(fp) + sum(fn) + sum(tn))
    precision = [tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0 for i in range(n_classes)]
    recall = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0 for i in range(n_classes)]
    specificity = [tn[i] / (fp[i] + tn[i]) if (fp[i] + tn[i]) > 0 else 0.0 for i in range(n_classes)]
    f1_score = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0.0 for i in range(n_classes)]
    roc_auc = [roc_auc_score(targets[i, :].detach().cpu().numpy(), predictions[i, :].detach().cpu().numpy()) for i in range(n_classes)]

    visualize_ROC = False
    if visualize_ROC:
        # Create a figure and axes
        fig, ax = plt.subplots()

        # Plot ROC curve for each class
        for i in range(len(roc_auc)):
            ax.plot([0, 1], [0, 1], 'k--')  # Plot diagonal line
            ax.plot(1 - specificity[i], recall[i], label='Class {} (AUC = {:.2f})'.format(i, roc_auc[i]))

        # Set labels and title
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')

        # Set legend
        ax.legend()
        # Show the plot
        plt.show()

    # precision, recall (or sensitivity), F1-score, roc_auc
    return precision, recall, f1_score, roc_auc, MI_size_pre, MI_size_gd, center_distance, AHA_loc_score

def calculate_chamfer_distance_old(x, y):
    """
    Computes the Chamfer distance between two point clouds.

    Args:
        x: Tensor of shape (n_batch, n_point, n_label).
        y: Tensor of shape (n_batch, n_point, n_label).

    Returns:
        chamfer_distance: Tensor of shape (1,)
    """
    x_expand = x.unsqueeze(2)  # Shape: (n_batch, n_point, 1, n_label)
    y_expand = y.unsqueeze(1)  # Shape: (n_batch, 1, n_point, n_label)
    diff = x_expand - y_expand
    dist = torch.sum(diff**2, dim=-1)  # Shape: (n_batch, n_point, n_point)
    dist_x2y = torch.min(dist, dim=2).values  # Shape: (n_batch, n_point)
    dist_y2x = torch.min(dist, dim=1).values  # Shape: (n_batch, n_point)
    chamfer_distance = torch.mean(dist_x2y, dim=1) + torch.mean(dist_y2x, dim=1)  # Shape: (n_batch,)
    return torch.mean(chamfer_distance)

def calculate_chamfer_distance(x, y):
    dist_x_y = torch.cdist(x, y)
    min_dist_x_y, _ = torch.min(dist_x_y, dim=1)
    min_dist_y_x, _ = torch.min(dist_x_y, dim=0)
    chamfer_distance = torch.mean(min_dist_x_y) + torch.mean(min_dist_y_x)

    return torch.mean(chamfer_distance)

def per_class_PCdist(pcd1, pcd2, dist_type='EDM', n_class=3):

    # Extract points from prediction and ground truth for each class
    LV_endo_pcd1, LV_epi_pcd1, RV_endo_pcd1 = torch.split(pcd1, n_class, dim=2)
    LV_endo_pcd2, LV_epi_pcd2, RV_endo_pcd2 = torch.split(pcd2, n_class, dim=2)

    # Note that ChamferDistance has O(n log n) complexity, while EMD has O(n2), which is too expensive to compute during training
    if dist_type == 'EDM':
        PCdist = calculate_emd
    else:
        PCdist = calculate_chamfer_distance
    LV_endo_loss = PCdist(LV_endo_pcd1, LV_endo_pcd2)
    LV_epi_loss = PCdist(LV_epi_pcd1, LV_epi_pcd2)
    RV_endo_loss = PCdist(RV_endo_pcd1, RV_endo_pcd2)
    combined_loss = (LV_endo_loss + LV_epi_loss + RV_endo_loss) / n_class

    return combined_loss

def calculate_emd(x1, x2, eps=1e-8, norm=1):
    """
    Calculates the Earth Mover's Distance (EMD) between two batches of point clouds.

    Args:
    - x1: A tensor of shape (batch_size, num_points, num_dims) representing the first batch of point clouds.
    - x2: A tensor of shape (batch_size, num_points, num_dims) representing the second batch of point clouds.
    - eps: A small constant added to the distance matrix to prevent numerical instability.
    - norm: The order of the norm used to calculate the distance matrix (default is L1 norm).

    Returns:
    - A tensor of shape (batch_size,) representing the EMD between each pair of point clouds in the batches.
    """
    batch_size, num_points, num_dims = x1.size()

    # Calculate distance matrix between points in each batch
    dist_mat = torch.cdist(x1, x2, p=norm)

    # Initialize flow matrix with zeros
    flow = torch.zeros(batch_size, num_points, num_points, requires_grad=True).to(x1.get_device())

    # Compute EMD using PyTorch's Sinkhorn algorithm
    for i in range(batch_size):
        flow[i] = F.sinkhorn_knopp(dist_mat[i], eps=eps)

    # Calculate total EMD for each pair of point clouds in the batches
    emd = torch.sum(flow * dist_mat, dim=(1, 2))

    return emd

def calculate_inference_loss(y_MI, gt_MI_label, mu, log_var, partial_input):
    PC_xyz = partial_input[:, 0:3, :]
    PC_tv =  torch.where((partial_input[:, 7, :] == 0.0) & (partial_input[:, 6, :] > 0), 1, 0)

    # x_input = partial_input[0].cpu().detach().numpy()
    # x_input_lab = PC_tv[0].cpu().detach().numpy().astype(int)
    # visualize_PC_with_label(x_input[0:3, :].transpose(), x_input_lab, filename='RNmap_pre.pdf')

    class_weights = torch.FloatTensor([1, 10, 10]).to(y_MI.get_device())
    loss_func_CE = nn.CrossEntropyLoss() # weight=class_weights

    y_MI_label = torch.argmax(y_MI, dim=1)
    loss_compactness, loss_MI_size, loss_MI_RVpenalty = calculate_MI_distribution_loss(y_MI_label, gt_MI_label, PC_xyz.permute(0, 2, 1), PC_tv)
    loss_CE = loss_func_CE(y_MI, gt_MI_label)
    Dice = calculate_Dice(y_MI, gt_MI_label, num_classes=3)
    loss_Dice = torch.sum((1.0-Dice) * class_weights)

    KL_loss = -0.5 * torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var))

    return loss_CE + 0.1*loss_Dice, loss_compactness, loss_MI_RVpenalty, loss_MI_size, KL_loss

def calculate_MI_distribution_loss(y_MI_label, gt_MI_label, PC_xyz, PC_tv):
    """
    计算点云数据的compactness
    
    Args:
        point_cloud: 点云数据，shape为(B, N, 3), only work when B=1
    
    Returns:
        compactness: 点云数据的compactness
    """
    y_MI_label_mask = torch.where((y_MI_label % 3) == 0, 0, 1).bool()
    gt_MI_label_mask = torch.where((gt_MI_label % 3) == 0, 0, 1).bool()
    
    compactness_sum = torch.tensor(0.0, requires_grad=True).to(y_MI_label.get_device())
    MI_size_div_sum = torch.tensor(0.0, requires_grad=True).to(y_MI_label.get_device())
    MI_RVpenalty_sum = torch.tensor(0.0, requires_grad=True).to(y_MI_label.get_device())

    num_iter = 0
    for i_batch in range(PC_xyz.shape[0]):
        y_PC_xyz_masked = PC_xyz[i_batch][y_MI_label_mask[i_batch]]
        gt_PC_xyz_masked = PC_xyz[i_batch][gt_MI_label_mask[i_batch]]
        
        if gt_PC_xyz_masked.shape[0]==0 or y_PC_xyz_masked.shape[0]==0:
            continue
        
        MI_size_div = abs(gt_PC_xyz_masked.size(0) - y_PC_xyz_masked.size(0))/gt_PC_xyz_masked.size(0)
        MI_size_div_sum = MI_size_div_sum.add(torch.tensor(MI_size_div, dtype=torch.float32).to(y_MI_label.get_device()))

        MI_RVpenalty = torch.sum(PC_tv[i_batch]*y_MI_label[i_batch])/y_PC_xyz_masked.shape[0]
        MI_RVpenalty_sum = MI_RVpenalty_sum.add(MI_RVpenalty)

        visual_check = False
        if visual_check:
            y_predict = y_MI_label_mask[i_batch].cpu().detach().numpy()
            x_input = PC_xyz[i_batch].cpu().detach().numpy()
            visualize_PC_with_label(x_input[y_predict], y_predict[y_predict], filename='RNmap_gd.jpg')
            visualize_PC_with_label(x_input, y_predict, filename='RNmap_pre.jpg')

        y_MI_center = torch.mean(y_PC_xyz_masked, dim=0).unsqueeze(0)
        gt_MI_center = torch.mean(gt_PC_xyz_masked, dim=0).unsqueeze(0)
        y_dist_sq = torch.sum((y_PC_xyz_masked - y_MI_center) ** 2, dim=1)
        gt_dist_sq = torch.sum((y_PC_xyz_masked - gt_MI_center) ** 2, dim=1)

        # max_distance = torch.max(torch.sqrt(torch.sum((PC_xyz[AHA_id>0][:, None] - PC_xyz[AHA_id>0]) ** 2, dim=2)))
        max_distance = torch.max(torch.sqrt(torch.sum((gt_PC_xyz_masked - gt_MI_center) ** 2, dim=1))) 
        y_compactness = torch.mean(torch.sqrt(y_dist_sq))/max_distance
        gt_compactness = torch.mean(torch.sqrt(gt_dist_sq))/max_distance 

        compactness_sum = compactness_sum.add(y_compactness + gt_compactness)
        num_iter += (num_iter + 1)
    if num_iter != 0:
        return compactness_sum/num_iter, MI_size_div_sum/num_iter, MI_RVpenalty_sum/num_iter
    else:
        return compactness_sum, MI_size_div_sum, MI_RVpenalty_sum

def calculate_Dice(inputs, target, num_classes):
    
    target_onehot = F.one_hot(target, num_classes).permute(0, 2, 1)
    
    eps = 1e-6
    intersection = torch.sum(inputs * target_onehot, dim=[0, 2])
    cardinality = torch.sum(inputs + target_onehot, dim=[0, 2])
    Dice = (2.0 * intersection + eps) / (cardinality + eps)
    
    return Dice

def one_hot_argmax(input_tensor):
    """
    This function takes a PyTorch tensor as input and returns a tuple of two tensors:
    - One-hot tensor: a binary tensor with the same shape as the input tensor, where the value 1
      is placed in the position of the maximum element of the input tensor and 0 elsewhere.
    - Argmax tensor: a tensor with the same shape as the input tensor, where the value is the index
      of the maximum element of the input tensor.
    """
    input_tensor = input_tensor.permute(0, 2, 1).squeeze(0)
    max_indices = torch.argmax(input_tensor, dim=1)
    one_hot_tensor = torch.zeros_like(input_tensor)
    one_hot_tensor.scatter_(1, max_indices.view(-1, 1), 1)
    
    return one_hot_tensor.permute(1, 0)

if __name__ == '__main__':

    pcs1 = torch.rand(10, 1024, 4)
    pcs2 = torch.rand(10, 1024, 4)



