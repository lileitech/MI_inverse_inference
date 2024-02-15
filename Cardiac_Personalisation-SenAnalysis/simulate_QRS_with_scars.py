import argparse
import os
import glob
import os
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import signal
from pyDOE import lhs
from SALib.sample import saltelli
import math
from cobiveco_custom import *
from ecg_functions import *
from cellular_models import mitchell_schaeffer
from geometry_functions import *
from propagation_models import EikonalDjikstraTet
from utils import F_mkdir, plot_ecgs_with_scars

class simulate_QRS_with_scars:
    # Lets not have default values inside class definitions
    def __init__(self, data_dir, djikstra_purkinje_max_path_len, frequency, freq_cut, geometric_data_dir, lead_names, max_len_ecg, max_len_qrs,
                 nb_continuous_params, nb_leads, purkinje_speed, subject_name, MI_type, verbose):

        if verbose:
            print('Initialising QRS simulation')
        # self.geometry = CardiacGeoTet(subject_name, data_dir=data_dir, geometric_data_dir=geometric_data_dir, verbose=True)
        clinical_ecg_raw_ori = np.genfromtxt(data_dir + geometric_data_dir + subject_name + '/' + subject_name + '_clinical_full_ecg_UKB_raw_8leads.csv', delimiter=',')
        clinical_ecg_raw = np.delete(clinical_ecg_raw_ori, 0, axis=1)
        self.propagation_model = EikonalDjikstraTet(data_dir, djikstra_purkinje_max_path_len, geometric_data_dir, nb_continuous_params, purkinje_speed, subject_name, MI_type, verbose)
        self.ecg = PseudoEcgTet(electrode_positions=self.propagation_model.geometry.electrode_positions, frequency=frequency,
                                freq_cut=freq_cut, lead_names=lead_names, max_len_ecg=max_len_ecg,
                                max_len_qrs=max_len_qrs, nb_leads=nb_leads,
                                nodes_xyz=self.propagation_model.geometry.nodes_xyz, reference_ecg=clinical_ecg_raw,
                                tetrahedrons=self.propagation_model.geometry.tetrahedrons,
                                tetrahedron_centers=self.propagation_model.geometry.tetrahedron_centers, verbose=verbose)
        # print('ADEU ADEU')
        self.clinical_ecg_normalised = self.ecg.preprocess_ecg(clinical_ecg_raw, filtering=True, normalise=True, zero_align=True)

    def simulate_eikonal_qrs(self, fibre_speed, transmural_speed, normal_speed, endo_dense_speed, endo_sparse_speed, root_node_meta_indexes):
        speed_params = [fibre_speed, transmural_speed, normal_speed, endo_dense_speed, endo_sparse_speed]
        lat = self.propagation_model.simulate_lat(self.propagation_model.pack_particle_from_params(speed_params, root_node_meta_indexes))
        return lat, self.ecg.calculate_ecg(vm=None, lat=lat, filtering=True, zero_align=True, normalise=True)

if __name__ == "__main__":
    MI_sizes = ['B1_large', 'B1_small']
    # MI_locations = ['A1', 'A2', 'A3', 'A4', 'B1_large', 'B1_small', 'B2', 'B3']
    # MI_transmural_extents = ['transmural', 'subendo']
    MI_locations = ['A3']
    MI_transmural_extents = ['transmural']
    MI_CV_d_scales = [[0.1, 0.5],[0.01, 0.05]]

    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='E:/OneDrive - Nexus365/Big_data_inference/meta_data/')
    # parser.add_argument('--datapath', type=str, default='/Users/lei/Library/CloudStorage/OneDrive-Nexus365/Big_data_inference/meta_data/')
    parser.add_argument('--djikstra_purkinje_max_path_len', type=int, default=400) 
    parser.add_argument('--max_len_qrs', type=int, default=500)
    parser.add_argument('--subject_name', type=str, default='1000268')# 'DTI004_coarse' 
    parser.add_argument('--MI_location', type=str, default='A3')
    parser.add_argument('--MI_transmural_extent', type=str, default='transmural') # subendo
    parser.add_argument('--MI_CV_d_scale', type=list, default=[0.1, 0.5])

    args = parser.parse_args()

    # General run settings:
    verbose = True
    frequency = 1000
    freq_cut = 150
    geometric_data_dir = 'UKB_clinical_data/' # 'geometric_data/' 
    lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    max_len_qrs = args.max_len_qrs
    max_len_ecg = max_len_qrs
    nb_continuous_params = 5
    nb_leads = 8
    purkinje_speed = 0.2    # cm/ms
    data_dir = args.datapath
    results_dir = data_dir + 'results/'

    # Eikonal parameters
    endo_dense_speed = 0.15
    endo_sparse_speed = 0.1
    fibre_speed = 0.065 # Taggart et al. (2000) 
    normal_speed = 0.048 # Taggart et al. (2000) 
    transmural_speed = 0.051 # Taggart et al. (2000) 
    nb_root_nodes = 20

    # subject_list = ['1030660', '1031466', '1032040', '1032124', '1034262', '1035446', '1036193']

    dataPath = args.datapath + 'UKB_clinical_data'
    datafile = sorted(glob.glob(dataPath + '/1*'))
    for subjectid in range(len(datafile)):
        if subjectid > 0:
             break
        # subject_name = args.subject_name
        subject_name = datafile[subjectid].replace(dataPath, '').replace('\\', '') 
        subject_name = '1000268'
        # if subject_name not in subject_list:
        #      continue     
        djikstra_purkinje_max_path_len = args.djikstra_purkinje_max_path_len 
        scar_scale_str = str(100*args.MI_CV_d_scale[0])
        bz_scale_str = str(100*args.MI_CV_d_scale[1])
        # MI_location = args.MI_location
        # MI_transmural_extent = args.MI_transmural_extent 
        savefold = data_dir + geometric_data_dir + subject_name     

        # if os.path.exists(savefold + '/' + subject_name + '_simulated_ATM_normal.csv'):  
        #     os.remove(savefold + '/' + subject_name + '_simulated_ATM_normal.csv')

        for MI_location_index in range(len(MI_locations)):
            for MI_transmural_extent_index in range(len(MI_transmural_extents)):
                MI_location = MI_locations[MI_location_index]
                MI_transmural_extent = MI_transmural_extents[MI_transmural_extent_index]
                MI_type = MI_location + '_' + MI_transmural_extent # 'normal'

                print('...' + subject_name + ' starts to simulate ECG with MI: ' + MI_location + ' and ' + MI_transmural_extent + '...')
                
                # Step 1: Generate dictionary of Mitchell Schaeffer cell APs at prescribed range of APDs.
                sim = simulate_QRS_with_scars(data_dir, djikstra_purkinje_max_path_len, frequency, freq_cut, geometric_data_dir, lead_names, max_len_ecg, max_len_qrs, nb_continuous_params, nb_leads, purkinje_speed, subject_name, MI_type, verbose)

                # Step 2: Simulate ECG using a specific set of parameter values
                root_node_meta_indexes = sim.propagation_model.select_random_root_nodes(nb_root_nodes)
                simulated_lat, simulated_ecg_normalised = sim.simulate_eikonal_qrs(fibre_speed, transmural_speed, normal_speed, endo_dense_speed, endo_sparse_speed, root_node_meta_indexes)
                
                # save simulated ECG
                # MI_type = 'normal'
                F_mkdir(savefold)
                # savedir = savefold + '/' + subject_name + '_simulated_ECG_' + MI_type + '.csv'
                # np.savetxt(savedir, np.squeeze(simulated_ecg_normalised), delimiter=',')

                savedir = savefold + '/' + subject_name + '_simulated_ATM_' + MI_type + '.csv'
                np.savetxt(savedir, np.squeeze(simulated_lat), delimiter=',')

        # Step 3: Visualise simulated QRS against clinical
        # plot_ecgs_with_scars(data_dir + geometric_data_dir, subject_name)
        # sim.ecg.plot_ecgs((simulated_ecg_normalised, sim.clinical_ecg_normalised), labels=['Simulated', 'Clinical'], save_dir=results_dir, subject_name=subject_name)

