import os
import numpy as np
from multiprocessing import Pool, cpu_count
from OccupancyInputCTCF.utils import cmap_utils as utils_s
from OccupancyInputCTCF.utils import plots as mplot

def generate_contact_maps(lefs_array, map_output_directory, str_frames, end_frame, every_frame, max_dist, res_convert, replication_number):
    """
    Generates contact maps using multiprocessing.
    Args:
        lefs_array (np.ndarray): Array of LEF positions.
        map_output_directory (str): Directory to save contact maps.
        str_frames (list): Starting frames for contact map calculation.
        end_frame (int): Ending frame for contact map calculation.
        every_frame (int): Interval between frames.
        max_dist (int): Maximum distance for contact map calculation.
        res_convert (int): Resolution conversion factor.
        replication_number (int): Number of replications.
    Returns:
        np.ndarray: Combined contact map.
    """
    print("Generating contact maps using multiprocessing...")

    # Create output directories for contact maps
    map_output_dirs = utils_s.create_contact_map_folders(len(str_frames), map_output_directory)

    # Worker function for multiprocessing
    def worker(map_output_dir, str_frame):
        utils_s.calculate_contact_map_save(
            lefs_array, str_frame, end_frame, every_frame, max_dist, res_convert, replication_number, map_output_dir
        )

    # Run multiprocessing
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(worker, zip(map_output_dirs, str_frames))

    # Combine contact maps from all directories
    w_map = []
    for dirs in map_output_dirs:
        file_path = os.path.join(dirs, 'contact_map.npz')
        with np.load(file_path) as data:
            cmap = data['contact_map']
            w_map.append(cmap)

    whole_map = np.sum(w_map, axis=0)
    print("Contact maps generated successfully.")
    return whole_map

def generate_plots(region, whole_chip, whole_chip_ctcf, whole_map, output_file):
    """
    Generates experimental and simulated maps.
    Args:
        region (str): Genomic region.
        whole_chip (np.ndarray): Combined chip data.
        whole_chip_ctcf (np.ndarray): Combined CTCF chip data.
        whole_map (np.ndarray): Combined contact map.
        output_file (str): Path to save the plot.
    """
    print("Step 8: Generating plots...")
    mplot.plot_chip_hic(region, whole_chip, whole_chip_ctcf, whole_map, res=200000, output_file=output_file)
    print("Step 8 complete. Output at", output_file)


def calculate_frip(paramdict, lefs_array, lst, window_size):
    """
    Calculates the fraction of extruder reads in peaks (FRiP).
    """
    print("Step 9: Calculating FRiP...")
    ### list of boundary elements on all replications
    rep = paramdict['number_of_replica'] 
    mon = paramdict['monomers_per_replica']
    site = paramdict['sites_per_monomer']
    mapN = paramdict['monomers_per_replica']*paramdict['sites_per_monomer']
    lst_t = []
    for i in range(rep):
        lst_t += list(np.array(lst)+i*mon*site)
    print(lst_t)
    lef_lefts = lefs_array[:,:,0].flatten()
    lef_rights = lefs_array[:,:,1].flatten()
    lef_positions = np.hstack((lef_lefts,lef_rights))
    peak_monomers = utils_s.peak_positions(lst_t, window_sizes = np.arange(-window_size, (window_size)+1))
    frip = utils_s.FRiP(mapN*rep, lef_positions, peak_monomers)
    print(f"FRiP is {frip}")
    return frip