import sys
import os
import json
import pandas as pd

# Add custom library path
sys.path.append('/home1/rahmanin/start/polychrom/projects/Site_wise_occupancy/OccupancyInputCTCF/')

# Import utility modules
import OccupancyInputCTCF.utils as util
import OccupancyInputCTCF.utils.snippet as snp
import OccupancyInputCTCF.utils.convert as convert
import OccupancyInputCTCF.utils.ml as ml
import OccupancyInputCTCF.utils.makeparams as params
import OccupancyInputCTCF.utils.One_d_simulation as simulation

# ================ INPUT PARAMETERS =====================
# Define genomic region and data files
region = 'chr1:34000000-34500000'
ctcf_motifs = '/project/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz'
ctcf_peaks = ('/project/fudenber_735/collaborations/karissa_2022/2022_09_features_for_RNAseq/ChIP-seq_in_WT-parental-E14' +
              '/CTCF_peaks_called_on_4reps_foundInatLeast2reps_noBlacklist.bed')

# Simulation parameters
with open('data/paramdict.json', 'r') as json_file:
    paramdict = json.load(json_file)

# Flags for simulation
run_md_sim = True  # Flag for MD simulation
use_predicted_occupancy = True  # Use predicted occupancy
input_occupancy = ctcf_peaks  # Input occupancy file

# =================== WORKFLOW ===========================
print("Starting workflow...")

# Step 1: Preprocessing
print("Step 1: Preprocessing...")
ctcf_bed_df = snp.get_region_snippet(ctcf_peaks, ctcf_motifs, region)
ctcf_bed_file = f'processing/{region}.csv'
ctcf_bed_df.to_csv(ctcf_bed_file, index=True)
print("Step 1 complete. Output:", ctcf_bed_df)

# Step 2: Predicting Occupancy Rate
print("Step 2: Predicting Occupancy Rate...")
#if use_predicted_occupancy:
#    ctcf_occup_bed_df = convert.make_ctcf_occupancy(ctcf_bed_df, frequency_df)
#else:
#    ctcf_occup_bed_df = convert.convert_ctcf_occupancy(ctcf_bed_df, ctcf_peaks)
#print("Step 2 complete. Output:", ctcf_occup_bed_df)

# Step 3: Refining Occupancy for Overlapping Sites
print("Step 3: Refining Occupancy...")
ctcf_occup_bed_df = pd.read_csv('processing/'+region+'_predicted_occupancy.csv')
refined_occupancy = convert.get_refined_occupancy(ctcf_occup_bed_df, region)
print("Step 3 complete. Output:", refined_occupancy)

# Step 4: Generating Barrier List with Occupancy and Lifetimes
print("Step 4: Generating Barrier List...")
CTCF_left_positions, CTCF_right_positions, ctcf_loc_list, ctcf_lifetime_list, ctcf_offtime_list = convert.get_ctcf_list(refined_occupancy, paramdict)
print("Step 4 complete. Right barriers:", CTCF_right_positions, "Left barriers:", CTCF_left_positions)

# Step 5: Running 1D Simulation
print("Step 5: Running 1D Simulation...")
output_path = 'simulations/sims/'
file_name = params.paramdict_to_filename(paramdict)
output_directory = output_path + 'folder_' + file_name.split('file_')[1]
os.makedirs(output_directory, exist_ok=True)
paramdict['monomers_per_replica'] = convert.get_lattice_size(region, lattice_site=250)
ctcf_params = convert.get_ctcf_list(refined_occupancy, paramdict)
trajectory_length = 10
lef_positions = simulation.Perform_1d_simulation(paramdict, ctcf_params, trajectory_length, output_directory)
print("Step 5 complete. Output at", output_path)

# Optional Step 6: MD Simulation
#if run_md_sim:
#    print("Step 6: Running optional MD simulation...")
    # md_simulation_results = run_md_simulation(parameters)  # Uncomment if applicable
#    print("Step 6 complete.")

#print("Workflow complete.")
