# Genomic region
REGION = 'chr1:36000000-37500000'

# File paths
PARAMDICT_PATH = 'data/paramdict.json'
OUTPUT_PATH = 'simulations/'
MAP_OUTPUT_DIRECTORY = './data/contact_maps/'

# Load parameters from JSON
import json
with open(PARAMDICT_PATH, 'r') as json_file:
    PARAMDICT = json.load(json_file)

# Experimental paramters
import OccupancyInputCTCF.utils.snippet as snp
import OccupancyInputCTCF.utils.experimental_path as exp
input_occupancy = exp.ctcf_peaks  # Input occupancy file
ctcf_bed_df = snp.get_region_snippet(exp.ctcf_peaks, exp.ctcf_motifs, REGION)

# Simulation parameters
TRAJECTORY_LENGTH = 1000
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
calculate_frip = False

# Md simulation parameters
PARAMDICT_md = {}
PARAMDICT_md['stiff']=1.5
PARAMDICT_md['dens']=0.2
PARAMDICT_md['saveEveryBlocks']=5
PARAMDICT_md['restartSimulationEveryBlocks']=100
PARAMDICT_md['steps'] = 20
import sys
import ast

filename = sys.argv[-1]

print('this is file name %s'%filename)

params = [ast.literal_eval(i) for i in filename.split('folder_')[1].split('_')[1::2]]
face, back, Clife, Cof, life, slife, birth, pause, sep, site, monomer, replica, steps, vel = params

PARAMDICT['LEF_lifetime']=[life]
PARAMDICT['LEF_stalled_lifetime']=[life]
PARAMDICT['LEF_separation']=sep
PARAMDICT['velocity_multiplier']=[vel]

print(PARAMDICT)

# Parameters for making maps from md simulations
import OccupancyInputCTCF.utils.convert as convert
import numpy as np
monomer_per_replica = convert.get_lattice_size(REGION, LATTICE_SITE)//10
mapN = 3 * monomer_per_replica #number of monomer to 
total_monomers = monomer_per_replica * REPLICATION_NUMBER
freq = 1 #frequent frames
mapstarts = (np.arange(0,total_monomers-2*monomer_per_replica , monomer_per_replica))
min_time = 200 # the number of steps to disregard when calculating contacts (for equilibration purpose)



