import bioframe

import numpy as np
import multiprocessing as mp
import os
from functools import partial

import networkx as nx
import time 
def create_lattice_graph(n, Lefs):
    G = nx.Graph()
    
    # Add regular lattice edges
    for i in range(n):
        if i + 1 < n:  # Right neighbor
            G.add_edge(i, i + 1, weight=1)
       # if i - 1 >= 0:  # Left neighbor
       #     G.add_edge(i, i - 1, weight=1)
    
    # Add loop connections
    for i, j in Lefs:
        G.add_edge(i, j, weight=1)
    
    return G
#G = create_lattice_graph(n, Lefs)
def closest_distance(G, start, end):
    try:
        return nx.shortest_path_length(G, source=start, target=end, weight="weight")
    except nx.NetworkXNoPath:
        return float('inf') 


def calculate_contact_map_save(lefs, str_frame, end_frame, every_frame, max_dist, res_convert, replication_number, output_dir):
    contact_map = []
    N = np.max(lefs) // res_convert + 1
    mod_i_values = mod_j_values = np.mod(np.arange(N // 10), N // 10)
    sites_p_r = N // 10
    
    for frame in range(str_frame, end_frame, every_frame):
        contact_matrix = np.zeros((sites_p_r, sites_p_r))
        lefs_t = lefs[frame, :, :] // res_convert
        start = time.time()

        G = create_lattice_graph(N, lefs_t)
        # Code to measure
        end = time.time()
        print(f"Elapsed time for graph: {end - start} seconds")
        start = time.time()
        for i in range(sites_p_r):
            for j in range(i + max_dist, sites_p_r):
                contact_matrix[i, j] = contact_matrix[j, i] = replication_number * (1 / (j - i) ** 1.5)
        
        for dupl in range(replication_number):
            start_idx = dupl * (N // 10)
            end_idx = (dupl + 1) * (N // 10)
            for i in range(start_idx, end_idx):
                for j in range(i + 1, end_idx):
                    if j < i + max_dist:
                        dist = closest_distance(G, i, j)
                        contact = 1 / (dist + 1) ** 1.5
                        contact_matrix[mod_i_values[i - start_idx], mod_j_values[j - start_idx]] += contact
                        contact_matrix[mod_j_values[j - start_idx], mod_i_values[i - start_idx]] += contact                  

        contact_map.append(contact_matrix)
        end = time.time()
        print(f"Elapsed time for contact map calculation: {end - start} seconds")
    np.savez_compressed(os.path.join(output_dir, 'contact_map.npz'), contact_map=np.sum(contact_map,axis=0))

def create_contact_map_folders(n, output_directory):
    for contact_id in range(1, n + 1):
        output_dirs= []
        file_name = f"contactmap_{contact_id}"
        output_directory_partial = os.path.join(output_directory, file_name)
        os.makedirs(output_directory_partial, exist_ok=True)
        output_dirs.append(output_directory_partial)
        return output_dirs

def region_data_frame(dataframe, region, lattice_size=250):
    """
    Extracts and processes a specified genomic region from a dataframe.

    Parameters:
    ----------
    dataframe : DataFrame
        The input dataframe containing genomic data with columns such as 'chrom', 'start', 'end', and 'mid'.
    region : str
        A string specifying the genomic region in the format 'chrom:start-end'.
    lattice_size : int, optional
        The size of each lattice or bin to segment the region, default is 250.

    Returns:
    -------
    DataFrame
        A dataframe filtered to the specified region with an added column 'lattice_loc' indicating
        the lattice location of each row based on 'mid' position and lattice size.
    """
    region_start = bioframe.parse_region_string(region)[1]
    region_dataframe = bioframe.select(dataframe, region, cols=['chrom', 'start', 'end'])
    region_dataframe['mid']=(region_dataframe.end+region_dataframe.start)/2
    region_dataframe['lattice_loc'] = ((region_dataframe['mid'] - region_start) // lattice_size).astype('int')
    region_dataframe = region_dataframe.reset_index(drop=True)
    return region_dataframe

def make_region_occupancy(file):
    df = pandas.read_csv(file)
    result_c = df.groupby(['lattice_loc', 'strand'])['predicted_occupancy'].apply(lambda x: 1-((1 - x).prod())).reset_index()
    result = result_c.merge(df.drop_duplicates(['lattice_loc', 'strand']), on=['lattice_loc', 'strand'], how='left')
    result = result.rename(columns={'predicted_occupancy_x':'predicted_occupancy'})
    result = result[['chrom','start','end','mid','strand','lattice_loc','predicted_occupancy']]
    return result 



def contact_map_from_lefs(dset, sites_per_replica):
    
    lef_array = np.mod(dset.reshape((-1, 2)), sites_per_replica)
    lef_array = lef_array[lef_array[:,1] > lef_array[:,0]]
    
    lef_map = np.histogram2d(lef_array[:,0], lef_array[:,1], np.arange(sites_per_replica))[0]
    
    return (lef_map + lef_map.T)

def chip_seq_from_lef(lef_positions, site_number_per_replica, min_time=0):
    lef_lefts = lef_positions[min_time:,:,0].flatten()
    lef_rights = lef_positions[min_time:,:,1].flatten() 
    lef_positions_aray = np.hstack((lef_lefts,lef_rights))
    hist,hist_ary = np.histogram(  np.mod( lef_positions_aray , site_number_per_replica ), np.arange(0,site_number_per_replica,1))
    return hist

def peak_positions(boundary_list, window_sizes=[1]):
    """
    Calculate peak positions based on a boundary_list within window_sizes.

    Args:
        boundary_list (list): List of boundary values.
        window_sizes (list, optional): List of window sizes. Defaults to [1].

    Returns:
        np.ndarray: Array containing peak positions.
    """
    peak_monomers = np.array([])

    for i in window_sizes:
        inds_to_add = [boundary + i for boundary in boundary_list]
        peak_monomers = np.hstack((peak_monomers, inds_to_add))

    return peak_monomers.astype(int)

def FRiP(num_sites_t, lef_positions, peak_positions ):
    
    hist,edges = np.histogram(  lef_positions  , np.arange(num_sites_t+1) )
    return np.sum(hist[peak_positions] )/len(lef_positions)

def chip_seq_from_ctcf(ctcf_array_right, ctcf_array_right_sites, ctcf_array_left, ctcf_array_left_sites, site_number_per_replica):
    ctcfrightary = np.concatenate([arr.flatten()*ctcf_array_right_sites for arr in ctcf_array_right if arr.size > 0])
    ctcfleftary = np.concatenate([arr.flatten()*ctcf_array_left_sites for arr in ctcf_array_left if arr.size >0])
    ctcfs = np.concatenate([ctcfrightary[ctcfrightary>0], ctcfleftary[ctcfleftary>0]])
    ctcfhist, hist_array = np.histogram(ctcfs, np.arange(0,site_number_per_replica,1))
    common_list = np.intersect1d(ctcf_array_right_sites, ctcf_array_left_sites)
    for elements in common_list:
        ctcfhist[elements] = ctcfhist[elements]/2
    return ctcfhist
