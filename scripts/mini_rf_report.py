#!/usr/bin/env python3
"""RF report for mini-backbone replicates.

Usage: mini_rf_report.py <mini_dir> <upham.tre> <variant1> <variant2> ...
Prints a table: rep x pair -> RF, RF_norm (via rf_distance.py logic).
"""
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location(
    'rf', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'rf_distance.py'))
rf = importlib.util.module_from_spec(spec)
sys.argv = ['rf_distance.py']  # swallow its argv use
spec.loader.exec_module(rf)

mini, upham, variants = sys.argv[1], sys.argv[2], sys.argv[3:]
pairs = [(v, 'upham') for v in variants] + \
        [(variants[i], variants[j])
         for i in range(len(variants)) for j in range(i + 1, len(variants))]

print(f'{"rep":<5}' + ''.join(f'{a}-{b:>15}' for a, b in pairs))
for rep in sorted(os.listdir(mini)):
    row = [rep]
    for va, vb in pairs:
        ta = upham if va == 'upham' else \
            os.path.join(mini, rep, va, 'tree', 'backbone.treefile')
        tb = upham if vb == 'upham' else \
            os.path.join(mini, rep, vb, 'tree', 'backbone.treefile')
        if not (os.path.exists(ta) and os.path.exists(tb)):
            row.append('            -')
            continue
        d, dn, _ = rf.rf_distance(ta, tb)
        row.append(f'{d:>6} ({dn:.3f})')
    print(f'{rep:<5}' + ''.join(f'{c:>16}' for c in row[1:]))
