import sys
import os

# Add custom library path
sys.path.append('/home1/rahmanin/start/polychrom/projects/Site_wise_occupancy/OccupancyInputCTCF/')

# Import utility modules
import utils.snippet as snp
import utils.convert as convert
import utils.ml as ml
import utils.makeparams as params
import utils.One_d_simulation as simulation

# ================ INPUT PARAMETERS =====================
# Define genomic region and data files
region = 'chr1:34000000-34500000'
ctcf_motifs = '/project/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz'
ctcf_peaks = ('/project/fudenber_735/collaborations/karissa_2022/2022_09_features_for_RNAseq/ChIP-seq_in_WT-parental-E14' +
              '/CTCF_peaks_called_on_4reps_foundInatLeast2reps_noBlacklist.bed')

# Simulation parameters
paramdict = {
    'CTCF_facestall': [1.0],
    'CTCF_backstall': [0.0],
    'CTCF_lifetime': [300],
    'CTCF_offtime': [180],
    'LEF_lifetime': [660],
    'LEF_stalled_lifetime': [660],
    'LEF_birth': [0.1],
    'LEF_pause': [0.9],
    'LEF_separation': 75,
    'sites_per_monomer': 10,
    'number_of_replica': 10,
    'velocity_multiplier': 1
}

# Flags for simulation
run_md_sim = True  # Flag for MD simulation
use_predicted_occupancy = True  # Use predicted occupancy
input_occupancy = ctcf_peaks  # Input occupancy file

# =================== WORKFLOW ===========================
print("Starting workflow...")

# Step 1: Preprocessing
print("Step 1: Preprocessing...")
ctcf_bed_df = snp.get_region_snippet(ctcf_peaks, ctcf_motifs, region)
print("Step 1 complete. Output:", ctcf_bed_df)

# Step 2: Predicting Occupancy Rate
print("Step 2: Predicting Occupancy Rate...")
predicted = False
if predicted:
    ctcf_occup_bed_df = convert.make_ctcf_occupancy(ctcf_bed_df, frequency_df)
else:
    ctcf_occup_bed_df = convert.convert_ctcf_occupancy(ctcf_bed_df, ctcf_peaks)
print("Step 2 complete. Output:", ctcf_occup_bed_df)

# Step 3: Refining Occupancy for Overlapping Sites
print("Step 3: Refining Occupancy...")
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
lef_positions = simulation.Perform_1d_simulation(paramdict, ctcf_params, 100, output_directory)
print("Step 5 complete. Output at", output_path)

# Optional Step 6: MD Simulation
if run_md_sim:
    print("Step 6: Running optional MD simulation...")
    # md_simulation_results = run_md_simulation(parameters)  # Uncomment if applicable
    print("Step 6 complete.")

print("Workflow complete.")
