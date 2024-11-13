import bioframe

import numpy as np

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

def axes_ary(region, res = 10000):
    region_ = bioframe.parse_region_string(region)
    ### array to show in the map
    mb = 1e6
    start_reg = region_[1]
    end_reg = region_[2]
    ary = list(np.arange(0, (end_reg-start_reg)//res+1, 100))
    ary_str = [region_[0]+':'+str(int(elements//mb))+' mb' for elements in np.arange(start_reg, end_reg+1,1e6)]
    return ary, ary_str

def make_region_occupancy(file):
    df = pandas.read_csv(file)
    result_c = df.groupby(['lattice_loc', 'strand'])['predicted_occupancy'].apply(lambda x: 1-((1 - x).prod())).reset_index()
    result = result_c.merge(df.drop_duplicates(['lattice_loc', 'strand']), on=['lattice_loc', 'strand'], how='left')
    result = result.rename(columns={'predicted_occupancy_x':'predicted_occupancy'})
    result = result[['chrom','start','end','mid','strand','lattice_loc','predicted_occupancy']]
    return result 