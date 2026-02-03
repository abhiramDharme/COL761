#!/usr/bin/env python3
"""Analyze why there are false negatives in the candidate generation."""

import numpy as np

db = np.load('db_features_muta.npy')
q = np.load('query_features_muta.npy')

print(f'DB shape: {db.shape}')
print(f'Query shape: {q.shape}')

# Check query 0
q0 = q[0]
print(f'\nQuery 0 features: {np.sum(q0 > 0)} non-zero features')
print(f'Query 0 non-zero indices: {np.where(q0 > 0)[0]}')

# Check how many DB graphs have qv <= gv
matches = 0
close_misses = 0
for gi in range(db.shape[0]):
    gv = db[gi]
    if np.all(q0 <= gv):
        matches += 1
    else:
        fails = np.sum(q0 > gv)
        if fails <= 3:
            close_misses += 1

print(f'\nQuery 0: {matches} candidates (qv <= gv)')
print(f'Close misses (<=3 failures): {close_misses}')

# Show which features cause most failures
failures_per_feature = np.zeros(db.shape[1])
for gi in range(db.shape[0]):
    failures_per_feature += (q0 > db[gi])
    
print(f'\nTop 10 problematic features for Q0:')
for fi in np.argsort(-failures_per_feature)[:10]:
    print(f'  Feature {fi}: {int(failures_per_feature[fi])} failures, q0[{fi}]={q0[fi]}')
