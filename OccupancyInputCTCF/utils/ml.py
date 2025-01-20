import pandas as pd
import pysam
import numpy as np
from .cnn_model import FlankCoreModel as CtcfOccupPredictor
import torch
import torch.nn.functional as F
torch.manual_seed(2024)
device = 'cpu' # prediction is fast, so GPU is not necessary

def pfm_to_pwm(pfm, background_frequency=0.25):
        s = pfm.sum(axis=0)
        pwm = np.log((pfm/s) / background_frequency)
        return pwm

def dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):
  """ 
  from basenji
    """
  if seq_len is None:
    seq_len = len(seq)
    seq_start = 0
  else:
    if seq_len <= len(seq):
      # trim the sequence
      seq_trim = (len(seq) - seq_len) // 2
      seq = seq[seq_trim:seq_trim + seq_len]
      seq_start = 0
    else:
      seq_start = (seq_len - len(seq)) // 2

  seq = seq.upper()

  # map nt's to a matrix len(seq)x4 of 0's and 1's.
  if n_uniform:
    seq_code = np.zeros((seq_len, 4), dtype='float16')
  else:
    seq_code = np.zeros((seq_len, 4), dtype='bool')
    
  for i in range(seq_len):
    if i >= seq_start and i - seq_start < len(seq):
      nt = seq[i - seq_start]
      if nt == 'A':
        seq_code[i, 0] = 1
      elif nt == 'C':
        seq_code[i, 1] = 1
      elif nt == 'G':
        seq_code[i, 2] = 1
      elif nt == 'T':
        seq_code[i, 3] = 1
      else:
        if n_uniform:
          seq_code[i, :] = 0.25
        elif n_sample:
          ni = random.randint(0,3)
          seq_code[i, ni] = 1

  return seq_code

def scan_sequence(pwm, pwm_rc, seq):
    k = pwm.shape[1]  # Length of the motif
    scores = np.array([
        np.nansum((pwm * seq[:, i:i+k]))
        for i in range(seq.shape[1] - k + 1)
    ])
    scores_rc = np.array([
        np.nansum((pwm_rc * seq[:, i:i+k]))
        for i in range(seq.shape[1] - k + 1)
    ])
    
    motif_position = scores.argmax()

    rc_motif_position = scores_rc.argmax()

    if scores[motif_position] > scores_rc[rc_motif_position]:
        return "+", motif_position
    else:
        return "-", rc_motif_position

def fetch_and_orient_from_fasta(bedfile, ref_genome_filepath='/project/fudenber_735/genomes/mm10/mm10.fa',
                          flanking_bp=15, core_bp=18):
    peaks_table = pd.read_table(bedfile, sep=',').iloc[:,:3]
    ref_genome = pysam.FastaFile(ref_genome_filepath)


    ctcf_pfm = np.loadtxt('data/MA0139.1.pfm', skiprows=1)
    ctcf_pwm = pfm_to_pwm(ctcf_pfm)

    ctcf_pfm_rc = np.flip(ctcf_pfm, axis=[0])
    ctcf_pfm_rc = np.flip(ctcf_pfm_rc, axis=[1])
    ctcf_pwm_rc = pfm_to_pwm(ctcf_pfm_rc)

    seqs = []
    for idx, row in peaks_table.iterrows():
        chrom, start, end = row[['chrom', 'start', 'end']]
        seq = dna_1hot(ref_genome.fetch(chrom, int(start), int(end)))
        seq = seq.T
        direction, ctcf_start = scan_sequence(ctcf_pwm, ctcf_pwm_rc, seq)
        seq = dna_1hot(ref_genome.fetch(chrom, int(start + ctcf_start - flanking_bp), int(start + ctcf_start + flanking_bp + core_bp)))
        seq = seq.T

        if direction == "-":
                seq = np.flip(seq, axis=[0])
                seq = np.flip(seq, axis=[1])
        seqs.append(seq)

    seqs = np.array(seqs)
    seq_len = 2*flanking_bp + core_bp

    return seqs, seq_len

def predict_ctcf_occupancy(ctcf_bed, model_weights_path='data/model_weights'):
    seqs, seq_len = fetch_and_orient_from_fasta(ctcf_bed)
    seqs = torch.tensor(seqs, dtype=torch.float32).to(device)
    peaks_table = pd.read_table(ctcf_bed, sep=',')

    weights = torch.load(model_weights_path, weights_only=True, map_location=torch.device(device))
    best_model = CtcfOccupPredictor(seq_len=seq_len,n_head=11, kernel_size=3).to(device)
    best_model.load_state_dict(weights)

    best_model.eval()
    with torch.no_grad():
        preds = best_model(seqs)
        preds = F.softmax(preds)

    peaks_table['predicted_occupancy'] = preds.numpy()[:,1]
    peaks_table.to_csv(f'{ctcf_bed}_with_predicted_occupancy', sep=',', index=False)
    
    return peaks_table