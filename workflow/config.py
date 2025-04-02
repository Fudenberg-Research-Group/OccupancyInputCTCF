# Genomic region
REGION = 'chr1:36000000-37500000'

import json

# Load parameters from JSON
with open('data/paramdict.json', 'r') as json_file:
    PARAMDICT = json.load(json_file)
    
# File paths
PARAMDICT_PATH = 'data/paramdict.json'
OUTPUT_PATH = 'simulations/sims_c/'
MAP_OUTPUT_DIRECTORY = './data/contact_maps/'

# Experimental paramters
import OccupancyInputCTCF.utils.snippet as snp
import OccupancyInputCTCF.utils.experimental_path as exp
input_occupancy = exp.ctcf_peaks  # Input occupancy file
ctcf_bed_df = snp.get_region_snippet(exp.ctcf_peaks, exp.ctcf_motifs, REGION)

# Simulation parameters
TRAJECTORY_LENGTH = 10000
N_SIMULATIONS = 5
LATTICE_SITE = 250
WINDOW_SIZE = 1

# Contact map parameters
START_FRAMES = [100, 200, 300, 400, 500]
END_FRAME_OFFSET = 800
EVERY_FRAME = 500
MAX_DIST = 150
RES_CONVERT = 40
REPLICATION_NUMBER = PARAMDICT['number_of_replica']

# Flags
RUN_MD_SIM = True
USE_PREDICTED_OCCUPANCY = True
