from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import random
import numpy as np
import os
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import glob 
from scipy.signal import find_peaks, peak_prominences
import pandas as pd
from myfunctions import dtw_ecg
import seaborn as sns
from collections import Counter
from fastdtw import fastdtw
import re
import shutil

def calculate_DTW(qrs_dur1, qrs_dur2):
    
    # Define a cost matrix that penalizes warping between points with different QRS durations
    def cost_matrix(x, y):
        qrs_penalty = 1  # Define a penalty for warping between points with different QRS durations
        qrs_cost = qrs_penalty * np.abs(x[0] - y[0])  # Calculate the QRS duration cost
        feat_cost = np.sum(np.abs(x[1:] - y[1:]))  # Calculate the feature cost
        return qrs_cost + feat_cost

    # Compute the DTW distance with QRS duration constraint
    distance, path = fastdtw(qrs_dur1[:, np.newaxis], qrs_dur2[:, np.newaxis], dist=cost_matrix)

    return distance
    
# Define a cost matrix that penalizes warping between points with different QRS durations
def cost_matrix(x, y):
    qrs_penalty = 1  # Define a penalty for warping between points with different QRS durations
    qrs_cost = qrs_penalty * np.abs(x[0] - y[0])  # Calculate the QRS duration cost
    feat_cost = np.sum(np.abs(x[1:] - y[1:]))  # Calculate the feature cost
    return qrs_cost + feat_cost

def calculate_poor_R_wave_progression(ecgs, ecg_name):
    n_signal = len(ecgs)
    n_lead = ecgs[0].shape[0]
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    eps = 0.001

    poor_R_wave_progression_list = {}
    for i_signal in range(n_signal):
        ecg_signal = ecgs[i_signal] 
        R_amplitude_list = {} 
        for i_lead in range(n_lead):
            ecg_signal_each_lead = ecg_signal[i_lead]
            peaks, _ = find_peaks(ecg_signal_each_lead, height=0)
            if len(peak_prominences(ecg_signal_each_lead, peaks)[0]) > 0:
                peak_R = peaks[np.argmax(peak_prominences(ecg_signal_each_lead, peaks)[0])] 
            else:
                peak_R = np.where(abs(ecg_signal_each_lead) < (np.min(abs(ecg_signal_each_lead)) + eps))[0]
            
            if not isinstance(peak_R, np.int64):
                if peak_R.shape[0] > 1:
                    peak_R = peak_R[1]
                else:
                    peak_R = peak_R[0]
            R_amplitude = ecg_signal_each_lead[peak_R]               
            R_amplitude_list[leadNames[i_lead]] = R_amplitude

        # print(R_amplitude_list['V4'])
        if (R_amplitude_list['V3'] > 2) or (R_amplitude_list['V4'] > 2):
            poor_R_wave_progression = 0
        elif R_amplitude_list['V4'] < R_amplitude_list['V3']:
            poor_R_wave_progression = 1
        elif R_amplitude_list['V3'] < R_amplitude_list['V2']:
            poor_R_wave_progression = 1
        elif R_amplitude_list['V2'] < R_amplitude_list['V1']:
            poor_R_wave_progression = 1
        else:
            poor_R_wave_progression = 1
        
        poor_R_wave_progression_list[ecg_name[i_signal]] = poor_R_wave_progression

    return poor_R_wave_progression_list

def calculate_QRS_duration_new(ecgs, ecg_name):
    n_signal = len(ecgs)
    n_lead = ecgs[0].shape[0]
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    QRS_duration_list = {}
    for i_signal in range(n_signal):
        ecg_signal = ecgs[i_signal] 
        R_amplitude_list = {} 
        for i_lead in range(n_lead):
            lead_signal = ecg_signal[i_lead]
            
            # locate onset and offset of QRS complexes
            onset_inds, offset_inds = locate_qrs_onset_offset(lead_signal, qrs_inds)
            
            # calculate QRS duration
            QRS_duration = (offset_inds - onset_inds) / 1000  # in seconds
            QRS_duration = ecgs[i_signal].shape[1]
            QRS_duration_list[ecg_name[i_signal]] = QRS_duration

    return QRS_duration_list

def calculate_QRS_duration(ecgs, ecg_name):
    n_signal = len(ecgs)

    QRS_duration_list = {}
    for i_signal in range(n_signal):
        QRS_duration = ecgs[i_signal].shape[1]
        QRS_duration_list[ecg_name[i_signal]] = QRS_duration

    return QRS_duration_list

def calculate_QRS_amplitude(ecgs, ecg_name, mesh_name):
    n_signal = len(ecgs)
    n_lead = ecgs[0].shape[0]
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    eps = 0.001

    QRS_amplitude_list_all_signal = {}
    for i_lead in range(n_lead):
        QRS_amplitude_list = {}
        for i_signal in range(n_signal):
            ecg_signal = ecgs[i_signal]  
            ecg_signal_each_lead = ecg_signal[i_lead]
            peaks, _ = find_peaks(ecg_signal_each_lead, height=0)
            if len(peak_prominences(ecg_signal_each_lead, peaks)[0]) > 0:
                peak_R = peaks[np.argmax(peak_prominences(ecg_signal_each_lead, peaks)[0])] 
            else:
                peak_R = np.where(abs(ecg_signal_each_lead) < (np.min(abs(ecg_signal_each_lead)) + eps))[0]
            peaks, _ = find_peaks(-ecg_signal_each_lead, height=0)
            if len(peak_prominences(-ecg_signal_each_lead, peaks)[0]) > 0:
                peak_Q = peaks[np.argmax(peak_prominences(-ecg_signal_each_lead, peaks)[0])]            
            else:
                peak_Q = np.where(abs(ecg_signal_each_lead) < (np.min(abs(ecg_signal_each_lead)) + eps))[0]
            
            if not isinstance(peak_R, np.int64):
                if peak_R.shape[0] > 1:
                    peak_R = peak_R[1]
                else:
                    peak_R = peak_R[0]

            if not isinstance(peak_Q, np.int64):
                if peak_Q.shape[0] > 1:
                    peak_Q = peak_Q[1]
                else:
                    peak_Q = peak_Q[0]
            
            QRS_amplitude = ecg_signal_each_lead[peak_R] - ecg_signal_each_lead[peak_Q]          
            QRS_amplitude_list[ecg_name[i_signal]] = QRS_amplitude
        QRS_amplitude_list_all_signal[leadNames[i_lead]] = QRS_amplitude_list

    pd_amplitude = pd.DataFrame.from_dict(QRS_amplitude_list_all_signal)
    pd_amplitude.insert(0, 'mesh_name', [mesh_name]*n_signal)

    return pd_amplitude

def calculate_pathological_Q(ecgs, ecg_name, mesh_name):
    n_signal = len(ecgs)
    n_lead = ecgs[0].shape[0]
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    eps = 1e-4

    Pathological_Q_list_all_signal = {}
    Q_R_amplitude_radio_list_all_signal = {}
    Q_duration_list_all_signal = {}
    for i_lead in range(n_lead):
        Pathological_Q_list = {}
        Q_R_amplitude_radio_list = {}
        Q_duration_list = {}
        for i_signal in range(n_signal):
            ecg_signal = ecgs[i_signal]  
            ecg_signal_each_lead = ecg_signal[i_lead]
            peaks, _ = find_peaks(ecg_signal_each_lead, height=0)
            if len(peak_prominences(ecg_signal_each_lead, peaks)[0]) > 0:
                peak_R = peaks[np.argmax(peak_prominences(ecg_signal_each_lead, peaks)[0])] 
            else:
                peak_R = np.where(abs(ecg_signal_each_lead) < (np.min(abs(ecg_signal_each_lead)) + eps))[0]
            peaks, _ = find_peaks(-ecg_signal_each_lead, height=0)
            if len(peak_prominences(-ecg_signal_each_lead, peaks)[0]) > 0:
                peak_Q = peaks[np.argmax(peak_prominences(-ecg_signal_each_lead, peaks)[0])]            
            else:
                peak_Q = np.where(abs(ecg_signal_each_lead) < (np.min(abs(ecg_signal_each_lead)) + eps))[0]
            
            if not isinstance(peak_R, np.int64):
                if peak_R.shape[0] > 1:
                    peak_R = peak_R[1]
                else:
                    peak_R = peak_R[0]

            if not isinstance(peak_Q, np.int64):
                if peak_Q.shape[0] > 1:
                    peak_Q = peak_Q[1]
                else:
                    peak_Q = peak_Q[0]

            R_amplitude = ecg_signal_each_lead[peak_R] 
            Q_amplitude = - ecg_signal_each_lead[peak_Q] 
            Q_R_amplitude_radio = abs(Q_amplitude/R_amplitude)

            # print(np.min(abs(ecg_signal_each_lead)))           
            signal_zero = (np.where(abs(ecg_signal_each_lead) < (np.min(abs(ecg_signal_each_lead)) + eps))[0]).tolist()
            Q_nearest_zero = min(signal_zero, key=lambda x: abs(x-peak_Q))
            Q_duration = abs(Q_nearest_zero - peak_Q)/1000

            visual_check = False
            if visual_check:
                plt.plot(ecg_signal_each_lead)
                plt.plot(peak_Q, ecg_signal_each_lead[peak_Q], "x")
                plt.plot(peak_R, ecg_signal_each_lead[peak_R], "o")
                # plt.plot(Q_nearest_zero, ecg_signal_each_lead[Q_nearest_zero], "o")
                plt.plot(np.zeros_like(ecg_signal_each_lead), "--", color="gray")
                plt.show()
            
            if Q_R_amplitude_radio > 0.25:
                Pathological_Q = 1
            elif Q_duration > 0.03:
                Pathological_Q = 1
            else:
                Pathological_Q = 0    

            Pathological_Q_list[ecg_name[i_signal]] = Pathological_Q
            Q_R_amplitude_radio_list[ecg_name[i_signal]] = Q_R_amplitude_radio
            Q_duration_list[ecg_name[i_signal]] = Q_duration

        Pathological_Q_list_all_signal[leadNames[i_lead]] = Pathological_Q_list
        Q_R_amplitude_radio_list_all_signal[leadNames[i_lead]] = Q_R_amplitude_radio_list
        Q_duration_list_all_signal[leadNames[i_lead]] = Q_duration_list


    pd_pathological_Q = pd.DataFrame.from_dict(Pathological_Q_list_all_signal)
    pd_pathological_Q.insert(0, 'mesh_name', [mesh_name]*n_signal)

    pd_Q_R_amplitude_radio = pd.DataFrame.from_dict(Q_R_amplitude_radio_list_all_signal)
    pd_Q_R_amplitude_radio.insert(0, 'mesh_name', [mesh_name]*n_signal)

    pd_Q_duration = pd.DataFrame.from_dict(Q_duration_list_all_signal)
    pd_Q_duration.insert(0, 'mesh_name', [mesh_name]*n_signal)

    return pd_pathological_Q, pd_Q_R_amplitude_radio, pd_Q_duration

def calculate_QRS_fractionation(ecgs, ecg_name, mesh_name):
    n_signal = len(ecgs)
    n_lead = ecgs[0].shape[0]

    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    QRS_fractionation_list_all_signal = {}
    for i_signal in range(n_signal):
        ecg_signal = ecgs[i_signal]
        QRS_fractionation_list = {}
        for i_lead in range(n_lead):
            leadName = leadNames[i_lead]
            ecg_signal_each_lead = ecg_signal[i_lead]
            peaks_R, _ = find_peaks(ecg_signal_each_lead, height=0.01) 
            peaks_Q, _ = find_peaks(-ecg_signal_each_lead, height=0.01)
            QRS_fractionation = peaks_R.shape[0] + peaks_Q.shape[0]
            if QRS_fractionation > 1:
                QRS_fractionation = QRS_fractionation - 3 # except Q, R, S peak
            else:
                QRS_fractionation = 0
            QRS_fractionation_list[leadName] = QRS_fractionation
        QRS_fractionation_list_all_signal[ecg_name[i_signal]] = QRS_fractionation_list
    
    pd_fractionation = pd.DataFrame.from_dict(QRS_fractionation_list_all_signal).transpose()
    pd_fractionation.insert(0, 'mesh_name', [mesh_name]*n_signal)
            
    return pd_fractionation

def calculate_QRS_dissimilarity(ecgs, ecg_name, mesh_name):  
    n_signal = len(ecgs)
    n_lead = ecgs[0].shape[0]
    leadNames = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
 
    QRS_dissimilarity_listlist = {}
    for i_signal in range(n_signal):
        if ecg_name[i_signal] == 'normal':
            continue
        ecg_signal = np.expand_dims(ecgs[i_signal], axis=0)
        dissimularity_score, dissimularity_score_lead = dtw_ecg(ecg_signal, ecgs[-1])
        QRS_dissimilarity_list = {}
        for i_lead in range(n_lead):
            QRS_dissimilarity_list[leadNames[i_lead]] = dissimularity_score_lead[i_lead]
        QRS_dissimilarity_listlist[ecg_name[i_signal]] = QRS_dissimilarity_list
    
    visual_peak = False
    if visual_peak:
        plt.figure(figsize=(15,10))
        pd_dissimilarity = pd.DataFrame.from_dict(QRS_dissimilarity_listlist).transpose()
        ax = sns.heatmap(pd_dissimilarity, annot=True, fmt=".1f", cmap=sns.cubehelix_palette(as_cmap=True))
        plt.savefig('QRS_dissimilarity.pdf')
        # plt.show()
    
    pd_dissimilarity = pd.DataFrame.from_dict(QRS_dissimilarity_listlist).transpose()
    pd_dissimilarity.insert(0, 'mesh_name', [mesh_name]*(n_signal-1))

    return pd_dissimilarity

def calculate_QRS_dissimilarity_matched(ecgs, ecg_name, mesh_name):  
    n_signal = len(ecgs)
 
    QRS_dissimilarity_avg_listlist = {}
    QRS_dissimilarity_max_listlist = {}
    
    for i_signal in range(n_signal):
        QRS_dissimilarity_avg_list = {}
        QRS_dissimilarity_max_list = {}
        for j_signal in range(n_signal):
            ecg_signal = np.expand_dims(ecgs[i_signal], axis=0)
            dissimularity_score_avg, dissimularity_score_lead = dtw_ecg(ecg_signal, ecgs[j_signal])
            QRS_dissimilarity_avg_list[ecg_name[j_signal]] = dissimularity_score_avg[0]
            QRS_dissimilarity_max_list[ecg_name[j_signal]] = max(dissimularity_score_lead)
        QRS_dissimilarity_avg_listlist[ecg_name[i_signal]] = QRS_dissimilarity_avg_list
        QRS_dissimilarity_max_listlist[ecg_name[i_signal]] = QRS_dissimilarity_max_list
    
    visual_peak = False
    if visual_peak:
        plt.figure(figsize=(20,16))
        # Getting the Upper Triangle of the co-relation matrix
        pd_mutual_dissimilarity_avg = pd.DataFrame.from_dict(QRS_dissimilarity_avg_listlist)
        matrix = np.flip(np.triu(pd_mutual_dissimilarity_avg), 0)
        sns.heatmap(pd_mutual_dissimilarity_avg, annot=True, mask=matrix, fmt=".1f", cmap=sns.cubehelix_palette(as_cmap=True)) # , vmin=0, vmax=25
        plt.savefig('QRS_mutual_dissimilarity_avg.pdf')

        plt.figure(figsize=(20,16))
        # Getting the lower Triangle of the co-relation matrix
        pd_mutual_dissimilarity_max = pd.DataFrame.from_dict(QRS_dissimilarity_max_listlist)
        matrix = np.flip(np.tril(pd_mutual_dissimilarity_max), 0)
        sns.heatmap(pd_mutual_dissimilarity_max, annot=True, mask=matrix, fmt=".1f", cmap="crest")
        plt.savefig('QRS_mutual_dissimilarity_max.pdf')
    
    pd_dissimilarity_avg = pd.DataFrame.from_dict(QRS_dissimilarity_avg_listlist)
    pd_dissimilarity_avg.insert(0, 'mesh_name', [mesh_name]*n_signal)

    pd_dissimilarity_max = pd.DataFrame.from_dict(QRS_dissimilarity_max_listlist)
    pd_dissimilarity_max.insert(0, 'mesh_name', [mesh_name]*n_signal)

    return pd_dissimilarity_avg, pd_dissimilarity_max

def ECG_rename(ECG_name):
    ECG_name = ECG_name.replace('A1', 'Septal').replace('A2', 'Apical').replace('A3', 'Ext anterior').replace('A4', 'Lim anterior')
    ECG_name = ECG_name.replace('B1', 'Lateral').replace('B2', 'Inferior').replace('B3', 'Inferolateral')
    # ECG_name = ECG_name.replace('normal', 'Baseline')
    ECG_name = ECG_name.replace('transmural', 'transmu')

    return ECG_name


if __name__ == '__main__':

    # datapath = '/Users/lei/Library/CloudStorage/OneDrive-Nexus365/Big_data_inference/meta_data/UKB_clinical_data/'
    datapath = 'E:/OneDrive - Nexus365/Big_data_inference/meta_data/UKB_clinical_data'
    datafile = sorted(glob.glob(datapath + '/1*'))
    QRS_duration_list_all = {}
    poor_R_wave_progression_list_all = {}
    
    for subjectid in range(len(datafile)):
        # if subjectid > 1:
        #      break
        meshName = datafile[subjectid].replace(datapath, '').replace('\\', '')
        print(meshName)
        # meshName = '1037159'
        signal_files = sorted(glob.glob(datapath + '/' + meshName + '/*_simulated_ECG*.csv'))
        ECG_list = list()
        ECG_name_list = list()

        num_signal = len(signal_files)
        for id in range(num_signal):   
            # if id > 0:
            #     continue                
            ECG_name = signal_files[id].replace(datapath + '/' + meshName, '').replace('\\', '').replace(meshName + '_simulated_ECG_', '').replace('.csv', '').replace('_', ' + ')
            # ECG_name = signal_files[id].replace(datapath + '/' + meshName + '/' + meshName + '_simulated_ECG_', '').replace('.csv', '').replace('_', ' + ')
            ECG_name = ECG_rename(ECG_name)
            
            # if ECG_name == 'Lateral + large + transmu + slow':
            #     continue
            if not re.compile(r'Apical|normal', re.IGNORECASE).search(ECG_name): 
                continue
            if ECG_name == 'Apical + subendo' or ECG_name == 'Apical + transmu':
                continue
            # print(ECG_name)

            ECG_value = np.loadtxt(signal_files[id], delimiter=',')
            ECG_list.append(ECG_value)
            ECG_name_list.append(ECG_name)

        pd_mutual_dissimilarity_avg_each, pd_mutual_dissimilarity_max_each = calculate_QRS_dissimilarity_matched(ECG_list, ECG_name_list, meshName) 
        pd_pathological_Q_each, pd_Q_R_amplitude_radio_each, pd_Q_duration_each = calculate_pathological_Q(ECG_list, ECG_name_list, meshName)  
        pd_dissimilarity_each = calculate_QRS_dissimilarity(ECG_list, ECG_name_list, meshName) 
        pd_fractionation_each = calculate_QRS_fractionation(ECG_list, ECG_name_list, meshName) 
        QRS_duration_each = calculate_QRS_duration(ECG_list, ECG_name_list)
        QRS_duration_list_all[meshName] = QRS_duration_each

        poor_R_wave_progression_each = calculate_poor_R_wave_progression(ECG_list, ECG_name_list)
        poor_R_wave_progression_list_all[meshName] = poor_R_wave_progression_each

        if subjectid == 0:
            pd_pathological_Q = pd_pathological_Q_each
            pd_Q_R_amplitude_radio = pd_Q_R_amplitude_radio_each
            pd_Q_duration = pd_Q_duration_each
            pd_dissimilarity = pd_dissimilarity_each
            pd_mutual_dissimilarity_avg = pd_mutual_dissimilarity_avg_each
            pd_mutual_dissimilarity_max = pd_mutual_dissimilarity_max_each
            pd_fractionation = pd_fractionation_each
        else:
            pd_pathological_Q = pd.concat([pd_pathological_Q_each, pd_pathological_Q])
            pd_Q_R_amplitude_radio = pd.concat([pd_Q_R_amplitude_radio_each, pd_Q_R_amplitude_radio])
            pd_Q_duration = pd.concat([pd_Q_duration_each, pd_Q_duration])
            pd_dissimilarity = pd.concat([pd_dissimilarity_each, pd_dissimilarity])
            pd_mutual_dissimilarity_avg = pd.concat([pd_mutual_dissimilarity_avg_each, pd_mutual_dissimilarity_avg])
            pd_mutual_dissimilarity_max = pd.concat([pd_mutual_dissimilarity_max_each, pd_mutual_dissimilarity_max])
            pd_fractionation = pd.concat([pd_fractionation_each, pd_fractionation])

 
    pd_mutual_dissimilarity_avg.index.name = 'Scenario name'
    pd_mutual_dissimilarity_avg = pd_mutual_dissimilarity_avg.groupby('Scenario name').mean()
    pd_mutual_dissimilarity_avg = pd_mutual_dissimilarity_avg.reindex(index = list(reversed(ECG_name_list)))
    pd_mutual_dissimilarity_avg.to_csv('QRS_mutual_dissimilarity_avg.csv', index=True)  

    pd_mutual_dissimilarity_max.index.name = 'Scenario name'
    pd_mutual_dissimilarity_max = pd_mutual_dissimilarity_max.groupby('Scenario name').mean()
    pd_mutual_dissimilarity_max = pd_mutual_dissimilarity_max.reindex(index = list(reversed(ECG_name_list)))
    pd_mutual_dissimilarity_max.to_csv('QRS_mutual_dissimilarity_max.csv', index=True)       

    pd_pathological_Q.index.name = 'Scenario name'
    pd_pathological_Q = pd_pathological_Q.groupby('Scenario name').mean()
    pd_pathological_Q = pd_pathological_Q.reindex(index = list(reversed(ECG_name_list)))
    pd_pathological_Q.to_csv('pathological_Q.csv', index=True)

    pd_Q_R_amplitude_radio.index.name = 'Scenario name'
    pd_Q_R_amplitude_radio = pd_Q_R_amplitude_radio.groupby('Scenario name').mean()
    pd_Q_R_amplitude_radio = pd_Q_R_amplitude_radio.reindex(index = list(reversed(ECG_name_list)))
    pd_Q_R_amplitude_radio.to_csv('Q_R_amplitude_radio.csv', index=True)

    pd_Q_duration.index.name = 'Scenario name'
    pd_Q_duration = pd_Q_duration.groupby('Scenario name').mean()
    pd_Q_duration = pd_Q_duration.reindex(index = list(reversed(ECG_name_list)))
    pd_Q_duration.to_csv('Q_duration.csv', index=True)

    pd_fractionation.index.name = 'Scenario name'
    pd_fractionation = pd_fractionation.groupby('Scenario name').mean()
    pd_fractionation = pd_fractionation.reindex(index = list(reversed(ECG_name_list)))
    pd_fractionation.to_csv('QRS_fractionation.csv', index=True)

    pd_dissimilarity.index.name = 'Scenario name'
    pd_dissimilarity = pd_dissimilarity.groupby('Scenario name').mean()
    pd_dissimilarity = pd_dissimilarity.reindex(index = list(reversed(ECG_name_list[:-1])))
    pd_dissimilarity.to_csv('QRS_dissimilarity.csv', index=True)

    pd_QRS_duration = pd.DataFrame.from_dict(QRS_duration_list_all)
    pd_QRS_duration.index.name = 'Scenario name'
    pd_QRS_duration = pd_QRS_duration.mean(axis=1, numeric_only=True).to_frame()
    pd_QRS_duration.set_axis(['QRS duration'], axis='columns', inplace=True)
    pd_QRS_duration.to_csv('QRS_duration.csv', index=True)

    pd_poor_R_wave_progression = pd.DataFrame.from_dict(poor_R_wave_progression_list_all)
    pd_poor_R_wave_progression.index.name = 'Scenario name'
    pd_poor_R_wave_progression = pd_poor_R_wave_progression.mean(axis=1, numeric_only=True).to_frame()
    pd_poor_R_wave_progression.set_axis(['Poor R wave progression'], axis='columns', inplace=True)
    pd_poor_R_wave_progression.to_csv('poor_R_wave_progression.csv', index=True)

    visual_QRS_mutual_dissimilarity = False
    if visual_QRS_mutual_dissimilarity:
        plt.figure(figsize=(20,16))
        # Getting the Upper Triangle of the co-relation matrix
        matrix = np.flip(np.triu(pd_mutual_dissimilarity_avg), 0)
        sns.heatmap(pd_mutual_dissimilarity_avg, annot=True, mask=matrix, fmt=".1f", square=True, cmap="Blues") # cmap=sns.cubehelix_palette(as_cmap=True)
        plt.savefig('fig_QRS_mutual_dissimilarity_avg.pdf')

        plt.figure(figsize=(20,16))
        # Getting the lower Triangle of the co-relation matrix
        matrix = np.flip(np.tril(pd_mutual_dissimilarity_max), 0) # GnBu
        sns.heatmap(pd_mutual_dissimilarity_max, annot=True, mask=matrix, fmt=".1f", square=True, cmap="GnBu") # crest
        plt.savefig('fig_QRS_mutual_dissimilarity_max.pdf')

    visual_QRS_dissimilarity = True
    if visual_QRS_dissimilarity:
        plt.figure(figsize=(10,8))
        # pd_dissimilarity = pd.read_csv('QRS_dissimilarity.csv')
        sns.heatmap(pd_dissimilarity, annot=True, fmt=".1f", square=True, cmap="crest")
        # plt.yticks(rotation=15)
        plt.savefig('fig_QRS_dissimilarity.pdf')

    visual_pathological_Q = False
    if visual_pathological_Q:
        plt.figure(figsize=(10,8))
        sns.heatmap(pd_pathological_Q, annot=True, fmt=".1f", cmap="Blues")
        plt.savefig('fig_pathological_Q.pdf')

        plt.figure(figsize=(10,8))
        sns.heatmap(pd_Q_R_amplitude_radio, annot=True, fmt=".1f", cmap="Blues")
        plt.savefig('fig_Q_R_amplitude_radio.pdf')

        plt.figure(figsize=(10,8))
        sns.heatmap(pd_Q_duration, annot=True, fmt=".2f", cmap="Blues")
        plt.savefig('fig_Q_duration.pdf')

    visual_QRS_fractionation = False
    if visual_QRS_fractionation:
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd_fractionation, annot=True, fmt=".1f", cmap="Blues")
        plt.savefig('fig_QRS_fractionation.pdf')

    visual_QRS_duration = False
    if visual_QRS_duration:
        plt.figure(figsize=(8,5))
        cols = ['#9C6D9D', '#E3A7BE', '#9C6D9D', '#E3A7BE', '#9C6D9D', '#E3A7BE', '#9C6D9D', '#E3A7BE', '#9C6D9D', '#E3A7BE', '#D9D3F2', '#EBD1DA', '#9C6D9D', '#E3A7BE', '#9C6D9D', '#E3A7BE', '#CDCDCD']
        sns.barplot(pd_QRS_duration, x=pd_QRS_duration.index, y='QRS duration', palette=cols)   
        # sns.barplot(pd_QRS_duration, x=pd_QRS_duration.index, y='QRS duration', capsize=.4, errcolor=".5", linewidth=3, edgecolor=".5", facecolor=(0, 0, 0, 0),)      
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('fig_QRS_duration.pdf')

    visual_poor_R_wave_progression = False
    if visual_poor_R_wave_progression:
        plt.figure(figsize=(8,5))
        cols = ['#9C6D9D', '#E3A7BE', '#9C6D9D', '#E3A7BE', '#9C6D9D', '#E3A7BE', '#9C6D9D', '#E3A7BE', '#9C6D9D', '#E3A7BE', '#D9D3F2', '#EBD1DA', '#9C6D9D', '#E3A7BE', '#9C6D9D', '#E3A7BE', '#CDCDCD']
        sns.barplot(pd_poor_R_wave_progression, x=pd_poor_R_wave_progression.index, y='Poor R wave progression', palette=cols)        
        # ax.set_facecolor('white')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('fig_poor_R_wave_progression.pdf')