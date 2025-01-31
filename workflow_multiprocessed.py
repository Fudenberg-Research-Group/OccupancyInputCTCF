import sys
import os
import json
import pandas as pd
import h5py
import numpy as np
from cooltools.lib.numutils import adaptive_coarsegrain
# Add custom library path
sys.path.append('/home1/rahmanin/start/polychrom/projects/Site_wise_occupancy/OccupancyInputCTCF/')

# Import utility modules
import OccupancyInputCTCF.utils as util
import OccupancyInputCTCF.utils.snippet as snp
import OccupancyInputCTCF.utils.plots as mplot
import OccupancyInputCTCF.utils.convert as convert
import OccupancyInputCTCF.utils.ml as ml
import OccupancyInputCTCF.utils.makeparams as params
import OccupancyInputCTCF.utils.One_d_simulation as simulation
import OccupancyInputCTCF.utils.experimental_path as exp
import OccupancyInputCTCF.utils.utils_s as utils_s
# ================ INPUT PARAMETERS =====================
# Define genomic region and data files
region = 'chr1:34850000-35000000'

# Simulation parameters
with open('data/paramdict.json', 'r') as json_file:
    paramdict = json.load(json_file)

# Flags for simulation
run_md_sim = True  # Flag for MD simulation
use_predicted_occupancy = True  # Use predicted occupancy
input_occupancy = exp.ctcf_peaks  # Input occupancy file

# =================== WORKFLOW ===========================
print("Starting workflow...")

# Step 1: Preprocessing
print("Step 1: Preprocessing...")
ctcf_bed_df = snp.get_region_snippet(exp.ctcf_peaks, exp.ctcf_motifs, region)
ctcf_bed_file = f'processing/{region}.csv'
ctcf_bed_df.to_csv(ctcf_bed_file, index=False)
print("Step 1 complete. Output:", ctcf_bed_df)

# Step 2: Predicting Occupancy Rate
print("Step 2: Predicting Occupancy Rate...")
if use_predicted_occupancy:
    ctcf_occup_bed_df = ml.predict_ctcf_occupancy(ctcf_bed_file)
else:
    ctcf_occup_bed_df = convert.convert_ctcf_occupancy(ctcf_bed_df, ctcf_peaks)
print("Step 2 complete. Output:", ctcf_occup_bed_df)

# Step 3: Refining Occupancy for Overlapping Sites
print("Step 3: Refining Occupancy...")
refined_occupancy = convert.get_refined_occupancy(ctcf_occup_bed_df, region)
print("Step 3 complete. Output:", refined_occupancy)

# Step 4: Generating Barrier List with Occupancy and Lifetimes
print("Step 4: Generating Barrier List...")
CTCF_left_positions, CTCF_right_positions, ctcf_loc_list, ctcf_lifetime_list, ctcf_offtime_list = convert.get_ctcf_list(
    refined_occupancy, paramdict)
print("Step 4 complete. Right barriers:", CTCF_right_positions, "Left barriers:", CTCF_left_positions)

# Step 5: Running 1D Simulation
import multiprocessing as mp
from functools import partial
print("Step 5: Running 1D Simulation...")
output_path = 'simulations/sims/'
trajectory_length = 1500
n = 5 # number of simulations in multiprocessing
file_name = params.paramdict_to_filename(paramdict)
output_directory = output_path + 'folder_' + file_name.split('file_')[1]
os.makedirs(output_directory, exist_ok=True)

paramdict['monomers_per_replica'] = convert.get_lattice_size(region, lattice_site=250)//10
ctcf_params = convert.get_ctcf_list(refined_occupancy, paramdict)
#lef_positions = simulation.Perform_1d_simulation(paramdict, ctcf_params, trajectory_length, output_directory)

outputt_dirs = []
for sim_id in range(1, n+1):
    file_name = f"simulation_{sim_id}"
    output_directory_partial = os.path.join(output_directory, f"{file_name}")
    os.makedirs(output_directory_partial, exist_ok=True)
    outputt_dirs.append(output_directory_partial)
    
def Perform_1d_simulation(output_directory, paramdict, ctcf_params, trajectory_length):
    return simulation.Perform_1d_simulation(paramdict, ctcf_params, trajectory_length, output_directory)
    
# use partial to set up some parameters of the function and leave output dir later
partially_set_Perform_1d_simulation = partial(Perform_1d_simulation, paramdict=paramdict, ctcf_params=ctcf_params, trajectory_length=trajectory_length)

def worker_init_fn():
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
with mp.Pool(processes=mp.cpu_count(), initializer=worker_init_fn) as pool:
    pool.map(
        partially_set_Perform_1d_simulation, outputt_dirs
    )
    
print("Step 5 complete. Output at", output_path)

# Optional Step 6: MD Simulation
if run_md_sim:
    print("Step 6: Running optional MD simulation...")
    #md_simulation_results = run_md_simulation(parameters)  # Uncomment if applicable
    print("Step 6 complete.")
    
### extracting maps from each simulaiton
# Step 7: Processing Outputs
print("Step 7: Processing Outputs...")
mapN = paramdict['monomers_per_replica']*paramdict['sites_per_monomer']
lattice_size = 0.25 # in kb
lef_array = []
n = 5
wmap = []
wchip = []
for sim_id in range(1, n+1):
    file_name = f"simulation_{sim_id}"
    output_directory_partial = os.path.join(output_directory, f"{file_name}")
    lef_file_path = os.path.join(output_directory_partial, 'LEFPositions.h5')
    if os.path.exists(lef_file_path):
        print(f"LEFPositions.h5 file found at {lef_file_path}.")
        lefs = h5py.File(lef_file_path, 'r')['positions']
        print('length of lefs is %s'%len(np.array(lefs)))
        chip = utils_s.chip_seq_from_lef(lefs, mapN)
        #lef_array.append(lefs)
        cmap = utils_s.contact_map_from_lefs(lefs[:], mapN)
        wmap.append(cmap)
        wchip.append(chip)
    else:
        print(f"Error: LEFPositions.h5 file not found at {lef_file_path}")


# Process and visualize contact map
whole_map = np.sum(wmap, axis=0)
whole_chip = np.sum(wchip, axis=0)
    
whole_map = cmap
#whole_map = np.sum(wmap, axis=0)
chip = utils_s.chip_seq_from_lef(lefs, mapN)

#cmap = utils_s.contact_map_from_lefs(lef_array[:],mapN)
cmap = whole_map
matrix = adaptive_coarsegrain(cmap+1, cmap+1, cutoff = 3, max_levels=3, min_shape = 2)
padded_matrix = np.pad(matrix, ((0, 1), (0, 1)), mode='constant')
dimension = 100 #dimension of the final matrix
binned_matrix = padded_matrix.reshape(dimension, len(padded_matrix)//dimension, dimension, len(padded_matrix)//dimension).sum(axis=(1, 3))
print("Step 7 complete. Output at", output_path)

# step 8: making maps 
print("Step8: making experimental and simulated maps ...")
output_file = 'newplots.pdf'
mplot.plot_chip_hic(region, whole_chip, binned_matrix, output_file=output_file)
print("Step 8 complete. Output at", output_file)

# step 9: calculating FRiP
print("Step 9: calculating fraction of extruder reads in peaks")
lst = np.concatenate((CTCF_right_positions , CTCF_left_positions))
### list of boundary elements on all replications
rep = paramdict['number_of_replica'] 
mon = paramdict['monomers_per_replica']
site = paramdict['sites_per_monomer']
lst_t = []
for i in range(rep):
    lst_t += list(np.array(lst)+i*mon*site)
print(lst_t)
lef_positions_t = []
window_size = 1
for sim_id in range(1, n+1):
    file_name = f"simulation_{sim_id}"
    output_directory_partial = os.path.join(output_directory, f"{file_name}")
    lef_file_path = os.path.join(output_directory_partial, 'LEFPositions.h5')
    if os.path.exists(lef_file_path):
        print(f"LEFPositions.h5 file found at {lef_file_path}.")
        lefs = h5py.File(lef_file_path, 'r')['positions']
        min_time = 100
        lef_lefts = lefs[min_time:,:,0].flatten()
        lef_rights = lefs[min_time:,:,1].flatten()
        lef_positions = np.hstack((lef_lefts,lef_rights))
        lef_positions_t.append(lef_positions)
    else:
        print(f"Error: LEFPositions.h5 file not found at {lef_file_path}")
#mapN=mon*site
lef_positions_concat = np.concatenate(lef_positions_t)
peak_monomers = utils_s.peak_positions(lst_t, window_sizes = np.arange(-window_size, (window_size)+1))
frip = utils_s.FRiP(mapN*rep, lef_positions_concat, peak_monomers)
print(f"Frip is {frip}")
print("Workflow complete.")
