#!/usr/bin/env python3
"""VBPI pilot driver for the PhylogenyAI backbone problem.

Loads a FASTA supermatrix + a support set of candidate trees (one Newick
per line, e.g. pruned .ufboot trees, FastTree runs, NJ perturbations),
builds the SBN support dictionary, trains the variational model, and
writes the highest-probability sampled topology.

Requires vbpi-torch (zcrabbit/vbpi-torch, unrooted/) on PYTHONPATH.

Usage:
  vbpi_driver.py --vbpi-src <vbpi-torch/unrooted> \
      --fasta sm.fasta --trees support.trees --out out_prefix \
      [--maxIter 5000] [--nParticle 10] [--device cuda] [--sites N]
"""
import argparse
import sys
import time
from collections import OrderedDict

p = argparse.ArgumentParser()
p.add_argument('--vbpi-src', required=True)
p.add_argument('--fasta', required=True)
p.add_argument('--trees', required=True, help='newick per line')
p.add_argument('--out', required=True)
p.add_argument('--maxIter', type=int, default=5000)
p.add_argument('--nParticle', type=int, default=10)
p.add_argument('--stepszTree', type=float, default=0.001)
p.add_argument('--stepszBranch', type=float, default=0.001)
p.add_argument('--nwarmStart', type=float, default=2000)
p.add_argument('--tf', type=int, default=500)
p.add_argument('--device', default='cuda')
p.add_argument('--sites', type=int, default=0, help='subsample sites (0=all)')
p.add_argument('--map-samples', type=int, default=100,
               help='trees sampled to pick MAP topology')
a = p.parse_args()

sys.path.insert(0, a.vbpi_src)
import numpy as np
import torch
from dataManipulation import loadData
from utils import get_support_from_mcmc
from vbpi import VBPI
from ete3 import Tree

dev = torch.device(a.device if torch.cuda.is_available()
                   or a.device == 'cpu' else 'cpu')
print(f'device: {dev}')

t0 = time.time()
data, taxa = loadData(a.fasta, 'fasta')
data = [s[:a.sites] if a.sites else s for s in data]
print(f'loaded {len(data)} taxa x {len(data[0])} sites ({time.time()-t0:.0f}s)')

t0 = time.time()
tree_dict, tree_names = OrderedDict(), []
for i, line in enumerate(open(a.trees)):
    line = line.strip()
    if line:
        tree_dict[f't{i}'] = Tree(line)
        tree_names.append(f't{i}')
rs_supp, ss_supp = get_support_from_mcmc(taxa, tree_dict, tree_names)
print(f'support: {len(tree_names)} trees -> {len(rs_supp)} rootsplits, '
      f'{sum(len(v) for v in ss_supp.values())} subsplits '
      f'({time.time()-t0:.0f}s)')
del tree_dict

model = VBPI(taxa, rs_supp, ss_supp, data, pden=np.ones(4) / 4.,
             subModel=('JC', 1.0), emp_tree_freq=None, psp=True).to(dev)
# plain-attribute tensors not moved by .to()
pm = model.phylo_model
pm.L = [x.to(dev) for x in pm.L]
pm.site_counts = pm.site_counts.to(dev) if torch.is_tensor(pm.site_counts) \
    else pm.site_counts
for k in ('pden', 'D', 'U', 'U_inv'):
    setattr(pm, k, getattr(pm, k).to(dev))
tm = model.tree_model
tm.ss_mask = tm.ss_mask.to(dev)
tm.update_CPDs()

model.learn({'tree': a.stepszTree, 'branch': a.stepszBranch},
            a.maxIter, test_freq=a.tf, n_particles=a.nParticle,
            warm_start_interval=a.nwarmStart,
            save_to_path=a.out + '.pt')

# MAP topology: argmax logq over sampled trees
best, best_q = None, -np.inf
with torch.no_grad():
    for _ in range(a.map_samples):
        t = model.tree_model.sample_tree()
        q = model.logq_tree(t).item()
        if q > best_q:
            best, best_q = t, q
best.write(format=1, outfile=a.out + '_map.tre')
print(f'MAP tree (logq={best_q:.2f}) -> {a.out}_map.tre')
