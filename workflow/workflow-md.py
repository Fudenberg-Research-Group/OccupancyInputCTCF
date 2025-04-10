import sys
import os
import json
from config import *
import pandas as pd
import h5py
import numpy as np
from analysis import calculate_frip
from cooltools.lib.numutils import adaptive_coarsegrain
import multiprocessing as mp
from functools import partial
# Add custom library path
sys.path.append('/home1/rahmanin/start/polychrom/projects/Site_wise_occupancy/OccupancyInputCTCF/')

# Import utility modules
import OccupancyInputCTCF.utils as util
import OccupancyInputCTCF.utils.plots as mplot
import OccupancyInputCTCF.utils.convert as convert
import OccupancyInputCTCF.utils.ml as ml
import OccupancyInputCTCF.utils.makeparams as params
import OccupancyInputCTCF.utils.One_d_simulation as simulation
import OccupancyInputCTCF.utils.md_simulation as mdsimulation
import OccupancyInputCTCF.utils.cmap_utils as utils_s
import warnings
warnings.filterwarnings('ignore')
import time
start = time.time()
# =================== WORKFLOW ===========================
print("Starting workflow...")

# Step 1: Preprocessing bed files in selected region
print("Step 1: Preprocessing...")
ctcf_bed_file = f'processing/{REGION}.csv'
ctcf_bed_df.to_csv(ctcf_bed_file, index=False)
print("Step 1 complete. Output:", ctcf_bed_df)

# Step 2: Predicting Occupancy Rate
print("Step 2: Predicting Occupancy Rate...")
if USE_PREDICTED_OCCUPANCY:
    ctcf_occup_bed_df = ml.predict_ctcf_occupancy(ctcf_bed_file)
else:
    ctcf_occup_bed_df = convert.convert_ctcf_occupancy(ctcf_bed_df, ctcf_peaks)
print("Step 2 complete. Output:", ctcf_occup_bed_df)

# Step 3: Refining Occupancy for Overlapping Sites
print("Step 3: Refining Occupancy...")
refined_occupancy = convert.get_refined_occupancy(ctcf_occup_bed_df, REGION)
print("Step 3 complete. Output:", refined_occupancy)

# Step 4: Generating Barrier List with Occupancy and Lifetimes
print("Step 4: Generating Barrier List...")
CTCF_left_positions, CTCF_right_positions, ctcf_loc_list, ctcf_lifetime_list, ctcf_offtime_list = convert.get_ctcf_list(
    refined_occupancy, PARAMDICT, insert_on='bound_time')
print("Step 4 complete. Right barriers:", CTCF_right_positions, "Left barriers:", CTCF_left_positions)

# Step 5: Running 1D Simulation
print("Step 5: Running 1D Simulation...")
file_name = params.paramdict_to_filename(PARAMDICT)
output_directory = OUTPUT_PATH + 'folder_' + file_name.split('file_')[1]
os.makedirs(output_directory, exist_ok=True)

PARAMDICT['monomers_per_replica'] = convert.get_lattice_size(REGION, lattice_site=250)//10
ctcf_params = convert.get_ctcf_list(refined_occupancy, PARAMDICT)

output_dirs = []
for sim_id in range(1, N_SIMULATIONS+1):
    file_name = f"simulation_{sim_id}"
    output_directory_partial = os.path.join(output_directory, f"{file_name}")
    os.makedirs(output_directory_partial, exist_ok=True)
    output_dirs.append(output_directory_partial)
    
def Perform_1d_simulation(output_directory, PARAMDICT, ctcf_params, TRAJECTORY_LENGTH):
    return simulation.Perform_1d_simulation(PARAMDICT, ctcf_params, TRAJECTORY_LENGTH, output_directory)
    
# use partial to set up some parameters of the function and leave output dir later
partially_set_Perform_1d_simulation = partial(Perform_1d_simulation, PARAMDICT=PARAMDICT, ctcf_params=ctcf_params, TRAJECTORY_LENGTH=TRAJECTORY_LENGTH)

def worker_init_fn():
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
with mp.Pool(processes=mp.cpu_count(), initializer=worker_init_fn) as pool:
    pool.map(
        partially_set_Perform_1d_simulation, output_dirs
    )
    
print("Step 5 complete. Output at", OUTPUT_PATH)

# Optional Step 6: MD Simulation
if RUN_MD_SIM:
    print("Step 6: Running optional MD simulation...")
    for sim_id in range(1, N_SIMULATIONS+1):
        file_name = f"simulation_{sim_id}"
        LEF_FILE_PATH = os.path.join(output_directory, f"{file_name}")
        mdsimulation.perform_md_simulation(LEF_FILE_PATH, PARAMDICT, PARAMDICT_md)
        cool_uri = output_directory + '/simulation_%s/map.mcool'%(sim_id)
        mdsimulation.make_md_contact_maps(mapN, mapstarts, freq, min_time, LEF_FILE_PATH, cool_uri)
    print("Step 6 complete.")
    
### extracting maps from each simulaiton
# Step 7: Processing Outputs
print("Step 7: Processing Outputs...")
MAP_OUTPUT_DIRECTORY = output_directory + '/contact_maps'
os.makedirs(MAP_OUTPUT_DIRECTORY, exist_ok=True)
mapN = PARAMDICT['monomers_per_replica']*PARAMDICT['sites_per_monomer']
lef_array, wmap, wchip, wchip_ctcf = [], [], [], []

for sim_id in range(1, N_SIMULATIONS+1):
    file_name = f"simulation_{sim_id}"
    output_directory_partial = os.path.join(output_directory, f"{file_name}")
    lef_file_path = os.path.join(output_directory_partial, 'LEFPositions.h5')
    if os.path.exists(lef_file_path):
        print(f"LEFPositions.h5 file found at {lef_file_path}.")
        lefs = h5py.File(lef_file_path, 'r')['positions']
        lef_array.append(lefs)
        print('length of lefs is %s'%len(np.array(lefs)))
        chip = utils_s.chip_seq_from_lef(lefs, mapN)
        cmap = utils_s.contact_map_from_lefs(lefs[:], mapN)
        ctcfchip = utils_s.chip_seq_from_ctcf(lef_file_path,mapN)
        wmap.append(cmap)
        wchip.append(chip)
        wchip_ctcf.append(np.array(ctcfchip))
    else:
        print(f"Error: LEFPositions.h5 file not found at {lef_file_path}")
lefs_array = np.vstack(lef_array)

# Multiprocessing for contact maps from graph based distance
map_output_dirs = utils_s.create_contact_map_folders(N_SIMULATIONS, MAP_OUTPUT_DIRECTORY+'/')
print(map_output_dirs)

end_frame = len(lefs_array)-END_FRAME_OFFSET
def worker(map_output_dir, START_FRAMES):
    utils_s.calculate_contact_map_save(lefs_array, START_FRAMES, end_frame, EVERY_FRAME, MAX_DIST, RES_CONVERT, REPLICATION_NUMBER, map_output_dir)

with mp.Pool(processes=mp.cpu_count()) as pool:
    pool.starmap(worker, zip(map_output_dirs, START_FRAMES))

w_map = []
for dirs in map_output_dirs:
    file_path = os.path.join(dirs, 'contact_map.npz')
    with np.load(file_path) as data:
        cmap = data['contact_map']  
        w_map.append(cmap)
whole_map = np.sum(w_map, axis=0)
whole_chip = np.sum(wchip, axis=0)
whole_chip_ctcf = np.sum(wchip_ctcf, axis=0)    
chip = utils_s.chip_seq_from_lef(lefs, mapN)
cmap = whole_map
print("Step 7 complete. Output at", OUTPUT_PATH)


# step 8: plots 
print("Step8: making experimental and simulated maps ...")
output_file = 'outputs/newplots_%s_trial_back_specials.pdf'%REGION
mplot.plot_chip_hic(REGION, whole_chip, whole_chip_ctcf, whole_map, res= 200000, output_file=output_file)
print("Step 8 complete. Output at", output_file)


# step 9: calculating FRiP
if calculate_frip:
    print("Step 9: calculating fraction of extruder reads in peaks")
    lst = np.concatenate((CTCF_right_positions , CTCF_left_positions))
    
    frip = calculate_frip(PARAMDICT, lefs_array, lst, WINDOW_SIZE)
    print(f"Frip is {frip}")
end = time.time()
workflow_time = end-start
print("Workflow complete. It took %s to complete that"%workflow_time)
