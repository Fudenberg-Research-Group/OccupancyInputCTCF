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
region = 'chr1:34000000-35000000'

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
print("Step 5: Running 1D Simulation...")
output_path = 'simulations/sims/'
trajectory_length = 15000
file_name = params.paramdict_to_filename(paramdict)
output_directory = output_path + 'folder_' + file_name.split('file_')[1]
os.makedirs(output_directory, exist_ok=True)
paramdict['monomers_per_replica'] = convert.get_lattice_size(region, lattice_site=250)//10
ctcf_params = convert.get_ctcf_list(refined_occupancy, paramdict)
lef_positions = simulation.Perform_1d_simulation(paramdict, ctcf_params, trajectory_length, output_directory)
print("Step 5 complete. Output at", output_path)

# Optional Step 6: MD Simulation
if run_md_sim:
    print("Step 6: Running optional MD simulation...")
    #md_simulation_results = run_md_simulation(parameters)  # Uncomment if applicable
    print("Step 6 complete.")
    
# step 7: processing, making simulated Hi-c maps and chip-seq profiles
print("Setp 7: processing outputs ...")
### making simulated maps
mapN = paramdict['monomers_per_replica']*paramdict['sites_per_monomer']
lattice_size = 0.25 # in kb
lef_array = h5py.File(output_directory+'/LEFPositions.h5','r')["positions"]
chip = utils_s.chip_seq_from_lef(lef_array, mapN)

cmap = utils_s.contact_map_from_lefs(lef_array[:],mapN)
matrix = adaptive_coarsegrain(cmap+1, cmap+1, cutoff = 3, max_levels=3, min_shape = 2)
padded_matrix = np.pad(matrix, ((0, 1), (0, 1)), mode='constant')
dimension = 100 #dimension of the final matrix
binned_matrix = padded_matrix.reshape(dimension, len(padded_matrix)//dimension, dimension, len(padded_matrix)//dimension).sum(axis=(1, 3))

print("Step 7 complete. Output at", output_path)

# step 8: making maps 
print("Step8: making experimental and simulated maps ...")
output_file = 'plots.pdf'
mplot.plot_chip_hic(region, chip, binned_matrix, output_file=output_file)
print("Step 8 complete. Output at", output_file)

print("Workflow complete.")
