# Path to the sample Hi-c map
bonev_file = '/project/fudenber_735/GEO/bonev_2017_GSE96107/distiller-0.3.1_mm10/results/coolers/HiC_ES.mm10.mapq_30.1000.mcool'
Li_file_wt = '/project2/fudenber_735/rahmanin/projects/Hi-C/results/coolers_library_group/WT.hg38.mapq_30.1000.mcool'
Li_file_nterms = '/project2/fudenber_735/rahmanin/projects/Hi-C/results/coolers_library_group/Y226A.hg38.mapq_30.1000.mcool'
# Path to the bigWig file
bw_path_rad21 = "/project/fudenber_735/GEO/justice_2020_GSE137272/mm10/GSE137272_Rad21-ChIPseq.bw"
bw_path_ctcf = "/project/fudenber_735/GEO/justice_2020_GSE137272/mm10/GSE137272_CTCF-ChIPseq.bw"
###importing ctcf peaks ####
ctcf_path='/project/fudenber_735/collaborations/karissa_2022/2022_09_features_for_RNAseq/ChIP-seq_in_WT-parental-E14'
### importing ctcf motifs ###
motif_directory='/project/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz'
### importing RAD21 
RAD21_path=ctcf_path+'/RAD21_peaks_called_on_6reps_foundInatLeast3reps_noBlacklist.bed'#.query(f'chrom in {clr.chromnames}')
ctcf_motifs = '/project/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz'
ctcf_peaks = ('/project/fudenber_735/collaborations/karissa_2022/2022_09_features_for_RNAseq/ChIP-seq_in_WT-parental-E14' +
              '/CTCF_peaks_called_on_4reps_foundInatLeast2reps_noBlacklist.bed')