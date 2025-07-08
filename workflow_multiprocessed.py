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
import OccupancyInputCTCF.utils.cmap_utils as utils_s
# ================ INPUT PARAMETERS =====================
# Define genomic region and data files
region = 'chr1:36000000-37500000'

# Simulation parameters
with open('data/paramdict.json', 'r') as json_file:
    paramdict = json.load(json_file)

pause_multip = round(1 / (1 - paramdict['LEF_pause'][0]), 1)

for key in ['CTCF_lifetime', 'CTCF_offtime', 'LEF_lifetime', 'LEF_stalled_lifetime']:
    val = paramdict[key]
    paramdict[key] = [x * pause_multip for x in val]

print(paramdict)

#paramdict['CTCF_backstall']=[0.3]
# Flags for simulation
run_md_sim = True  # Flag for MD simulation
use_predicted_occupancy = True  # Use predicted occupancy
input_occupancy = exp.ctcf_peaks  # Input occupancy file

# =================== WORKFLOW ===========================
print("Starting workflow...")

# Step 1: Preprocessing bed files in selected region
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
    refined_occupancy, paramdict, insert_on='bound_time')
print("Step 4 complete. Right barriers:", CTCF_right_positions, "Left barriers:", CTCF_left_positions)

# Step 5: Running 1D Simulation
import multiprocessing as mp
from functools import partial
print("Step 5: Running 1D Simulation...")
output_path = 'simulations/sims_c/'
trajectory_length = 1950
n = 5 # number of simulations in multiprocessing
file_name = params.paramdict_to_filename(paramdict)
output_directory = output_path + 'folder_' + file_name.split('file_')[1]
os.makedirs(output_directory, exist_ok=True)

paramdict['monomers_per_replica'] = convert.get_lattice_size(region, lattice_site=250)//10
ctcf_params = convert.get_ctcf_list(refined_occupancy, paramdict)
#lef_positions = simulation.Perform_1d_simulation(paramdict, ctcf_params, trajectory_length, output_directory)

output_dirs = []
for sim_id in range(1, n+1):
    file_name = f"simulation_{sim_id}"
    output_directory_partial = os.path.join(output_directory, f"{file_name}")
    os.makedirs(output_directory_partial, exist_ok=True)
    output_dirs.append(output_directory_partial)
    
def Perform_1d_simulation(output_directory, paramdict, ctcf_params, trajectory_length):
    return simulation.Perform_1d_simulation(paramdict, ctcf_params, trajectory_length, output_directory)
    
# use partial to set up some parameters of the function and leave output dir later
partially_set_Perform_1d_simulation = partial(Perform_1d_simulation, paramdict=paramdict, ctcf_params=ctcf_params, trajectory_length=trajectory_length)

def worker_init_fn():
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
with mp.Pool(processes=mp.cpu_count(), initializer=worker_init_fn) as pool:
    pool.map(
        partially_set_Perform_1d_simulation, output_dirs
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
lef_array = []
wmap = []
wchip = []
wchip_ctcf = []
for sim_id in range(1, n+1):
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
        wmap.append(cmap)
        wchip.append(chip)
        ctcf_array_right = np.array(h5py.File(lef_file_path, 'r')['CTCF_positions_right'])
        ctcf_array_left = np.array(h5py.File(lef_file_path, 'r')['CTCF_positions_left'])
        ctcf_array_right_sites = np.array(h5py.File(lef_file_path, 'r')['CTCF_sites_right'])
        ctcf_array_left_sites = np.array(h5py.File(lef_file_path, 'r')['CTCF_sites_left'])
        ctcfchip = utils_s.chip_seq_from_ctcf(ctcf_array_right, ctcf_array_right_sites, ctcf_array_left, ctcf_array_left_sites,mapN)
        wchip_ctcf.append(np.array(ctcfchip))
    else:
        print(f"Error: LEFPositions.h5 file not found at {lef_file_path}")
lefs_array = np.vstack(lef_array)

# Multiprocessing for contact maps
### creating approximated contact map from graph based distance
# Directory setup
map_output_directory = './data/contact_maps/'
map_output_dirs = utils_s.create_contact_map_folders(5, map_output_directory)

str_frames = [10, 20, 30, 40, 50]  # Different starting frames
end_frame = len(lefs_array)-80
every_frame = 50
max_dist = 150
res_convert = 40
replication_number = 10
def worker(map_output_dir, str_frame):
    utils_s.calculate_contact_map_save(lefs_array, str_frame, end_frame, every_frame, max_dist, res_convert, replication_number, map_output_dir)

with mp.Pool(processes=mp.cpu_count()) as pool:
    pool.starmap(worker, zip(map_output_dirs, str_frames))

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
print("Step 7 complete. Output at", output_path)


# step 8: plots 
print("Step8: making experimental and simulated maps ...")
output_file = 'outputs/newplots_%s_trial_back_.pdf'%region
mplot.plot_chip_hic(region, whole_chip, whole_chip_ctcf, whole_map, res= 200000, output_file=output_file)
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
