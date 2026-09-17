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
    applied = skipped = 0
    for old, new in subs:
        if old in s:
            s = s.replace(old, new)
            applied += 1
        else:
            assert new in s, f'{path}: pattern missing and patch absent: {old[:60]}'
            skipped += 1
    open(path, 'w').write(s)
    print(f'patched {path} ({applied} applied, {skipped} already done)')


patch(f'{d}/vector_sbnModel.py', [
    ("torch.ones(ss_len, dtype=torch.uint8)",
     "torch.ones(ss_len, dtype=torch.bool)"),
    ("masked_temp_mat = temp_mat.masked_fill(1-self.ss_mask, -float('inf'))",
     "masked_temp_mat = temp_mat.masked_fill(~self.ss_mask, -float('inf'))"),
    ("temp_mat = torch.zeros(self.ss_mask.size())",
     "temp_mat = self.CPD_params.new_zeros(self.ss_mask.size())"),
    ("self.one_tensor = torch.tensor([1.0])",
     "self.one_tensor = self.CPD_params.new_tensor([1.0])"),

    # Off-support fallback in sample_tree: sparse support sets (few trees)
    # leave parent subsplit contexts unobserved -> KeyError in
    # ss_reverse_map. Pool this clade's observed divisions from any sister
    # context; if never resolved, peel one taxon deterministically.
    ("    def sample_tree(self, rooted=False):",
     """    def _fallback_split(self, clade_bitarr):
        c = clade_bitarr.to01()
        cn = clade_bitarr.count()
        pool = [ch for k, chs in self.subsplit_supp_dict.items()
                if k[self.ntaxa:] == c for ch in chs
                if 0 < bitarray(ch).count() < cn]
        if pool:
            return pool[np.random.randint(len(pool))]
        b = bitarray('0' * self.ntaxa)
        b[clade_bitarr.find(1)] = 1
        return b.to01()

    def sample_tree(self, rooted=False):"""),
    ("""                split_prob = self.get_subsplit_CPDs(split_bitarr)
                # split = self.ss_reverse_map[split_bitarr][np.random.choice(len(split_prob), p=split_prob)]
                split = self.ss_reverse_map[split_bitarr][torch.multinomial(split_prob, 1).item()]""",
     """                if split_bitarr in self.ss_reverse_map:
                    split_prob = self.get_subsplit_CPDs(split_bitarr)
                    split = self.ss_reverse_map[split_bitarr][torch.multinomial(split_prob, 1).item()]
                else:
                    split = self._fallback_split(parent_clade_bitarr)"""),
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
