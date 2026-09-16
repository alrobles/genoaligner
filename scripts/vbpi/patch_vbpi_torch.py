#!/usr/bin/env python3
"""Patch vbpi-torch (zcrabbit) unrooted/ for modern torch + GPU device.

Fixes:
  - uint8 -> bool masks (masked_scatter_/masked_fill API change)
  - CPU-hardcoded tensor creation -> param-device tensors
Usage: patch_vbpi_torch.py <vbpi-torch/unrooted dir>
"""
import re
import sys

d = sys.argv[1]


def patch(path, subs):
    s = open(path).read()
    for old, new in subs:
        assert old in s, f'{path}: pattern not found: {old[:60]}'
        s = s.replace(old, new)
    open(path, 'w').write(s)
    print(f'patched {path}')


patch(f'{d}/vector_sbnModel.py', [
    ("torch.ones(ss_len, dtype=torch.uint8)",
     "torch.ones(ss_len, dtype=torch.bool)"),
    ("masked_temp_mat = temp_mat.masked_fill(1-self.ss_mask, -float('inf'))",
     "masked_temp_mat = temp_mat.masked_fill(~self.ss_mask, -float('inf'))"),
    ("temp_mat = torch.zeros(self.ss_mask.size())",
     "temp_mat = self.CPD_params.new_zeros(self.ss_mask.size())"),
    ("self.one_tensor = torch.tensor([1.0])",
     "self.one_tensor = self.CPD_params.new_tensor([1.0])"),
])

patch(f'{d}/base_branchModel.py', [
    ("self.feature_padded = torch.cat((self.sx, torch.zeros(1, self.feature_dim)), dim=0)",
     "self.feature_padded = torch.cat((self.sx, self.sx.new_zeros(1, self.feature_dim)), dim=0)"),
    ("neigh_ss_idxes = torch.LongTensor(neigh_ss_idxes)",
     "neigh_ss_idxes = torch.as_tensor(neigh_ss_idxes, dtype=torch.long, device=self.sx.device)"),
    ("branch_idx_map = torch.sort(torch.LongTensor(idx_list), dim=0, descending=False)[1]",
     "branch_idx_map = torch.sort(torch.as_tensor(idx_list, dtype=torch.long, device=self.sx.device), dim=0, descending=False)[1]"),
    ("samp_log_branch = torch.randn(n_particles, 2*self.ntips-3)",
     "samp_log_branch = torch.randn(n_particles, 2*self.ntips-3, device=self.sx.device)"),
])
print('done')
