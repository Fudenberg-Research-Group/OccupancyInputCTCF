import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import bioframe
import cooler
import pyBigWig
import OccupancyInputCTCF.utils.experimental_path as exp
import cooltools
from cooltools.lib.plotting import *


### importing Hi-C maps 
clr = cooler.Cooler(exp.bonev_file+'::resolutions/10000')
### importing RAD21 & CTCF peaks/motifs
RAD21=bioframe.read_table(exp.RAD21_path, schema='bed')
ctcf = bioframe.read_table(exp.ctcf_peaks, schema='bed')
motif=bioframe.read_table(exp.motif_directory)

ctcf['mid']=(ctcf.end+ctcf.start)/2
RAD21['mid']=(RAD21.end+RAD21.start)/2

bw_rad21 = pyBigWig.open(exp.bw_path_rad21)
bw_ctcf = pyBigWig.open(exp.bw_path_ctcf)

def plot_chip_hic(region, chip, binned_matrix, output_file='plots.png'):
    """
    Hi-C and ChIP-seq plots along with simulated data visualizations.
    
    Parameters:
    - region (str): Genomic region to visualize (e.g., 'chr6:50,000,000-51,000,000').
    - binned_matrix (ndarray): Simulated Hi-C matrix.
    - chip (ndarray): Simulated 1D extruder data.
    - output_file (str): Path to save the output plot.
    """

    region_ = bioframe.parse_region_string(region)
    chrom, start_reg, end_reg = region_

    values_rad21 = np.nan_to_num(bw_rad21.values(chrom, start_reg, end_reg))
    values_ctcf = np.nan_to_num(bw_ctcf.values(chrom, start_reg, end_reg))

    # Set up plot layout
    plt_width = 4
    f, axs = plt.subplots(
        figsize=(plt_width + plt_width + 2, plt_width + 5),
        ncols=4,
        nrows=3,
        gridspec_kw={'height_ratios': [4, 1, 1], "wspace": 0.01, 'width_ratios': [1, 0.04, 1, 0.04]},
        constrained_layout=True
    )

    norm = LogNorm(vmax=0.1)

    # Raw Hi-C data
    ax = axs[0, 0]
    im = ax.matshow(
        clr.matrix().fetch(region),
        norm=norm,
        cmap='fall'
    )
    ax.set_title(f'{chrom}:{start_reg:,}-{end_reg:,}')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    cax = axs[0, 1]
    plt.colorbar(im, cax=cax, label='raw counts')

    # RAD21 ChIP-seq
    ax1 = axs[1, 0]
    ax1.plot(np.arange(start_reg, end_reg), values_rad21)
    ax1.set_xlim([start_reg, end_reg])
    ax1.set_xlabel('position, bins')
    ax1.set_title('RAD21')
    axs[1, 1].set_visible(False)

    # CTCF ChIP-seq
    ax1 = axs[2, 0]
    ax1.plot(np.arange(start_reg, end_reg), values_ctcf, color='green', label='CTCF ChIP-seq')
    ax1.set_xlim([start_reg, end_reg])
    ax1.set_xlabel('position, bins')
    ax1.set_title('CTCF')
    axs[2, 1].set_visible(False)

    # Simulated Hi-C map
    ax = axs[0, 2]
    im = ax.matshow(np.log10(binned_matrix + 1), cmap='fall')
    ax.set_title('1D Simulation')
    cax = axs[0, 3]
    plt.colorbar(im, cax=cax, label='raw counts')

    # Simulated extruder data
    ax1 = axs[1, 2]
    ax1.plot(np.arange(len(chip)) * lattice_size, chip / np.sum(chip))
    ax1.set_xlabel('extruders position, bp')
    ax1.set_title('Simulation, Extruder')
    axs[1, 3].set_visible(False)

    ax1 = axs[2, 2]
    ax1.plot(np.arange(len(chip)) * lattice_size, chip / np.sum(chip), color='red')
    ax1.set_xlabel('extruders position')
    ax1.set_title('Simulation')
    axs[2, 3].set_visible(False)

    # Save and show plot
    plt.savefig(output_file)
    plt.show()
