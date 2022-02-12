import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
file_path = 'files/'
from tqdm import tqdm
import pickle
import concurrent.futures
from scipy.stats import pearsonr, spearmanr

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'depth_norm':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('depth_norm')]
    import sys
    sys.path.append(str(cwd))


depths = pickle.load(open(cwd / 'read_depth_distribution' / 'results' / 'depths.pkl', 'rb'))
probabilities, sample_df = pickle.load(open(cwd / 'read_depth_normalization' / 'results' / 'probabilities.pkl', 'rb'))

def get_fraction(sample):
    normed = []
    bin = sample_df.loc[sample_df['Tumor_Sample_Barcode'] == sample]['bin'].values[0]
    for i in range(100):
        count = 0
        total = 0
        while count < 100:
            total += 1
            if np.random.random() > depths[sample][2]:
                depth = np.random.choice(depths[sample][0], size=1, p=depths[sample][1] / sum(depths[sample][1]))
                if depth >= 8:
                    if np.random.random() > probabilities[bin][int(depth)]:
                        count += 1
            else:
                count += 1
        normed.append(total)
    return 100 / np.mean(normed)

fractions = {}
sample_df = sample_df[sample_df['Tumor_Sample_Barcode'].isin(depths)]
with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
    for sample, result in tqdm(zip(sample_df['Tumor_Sample_Barcode'].values, executor.map(get_fraction, sample_df['Tumor_Sample_Barcode'].values))):
        fractions[sample] = result

sample_df['fraction'] = sample_df['Tumor_Sample_Barcode'].apply(lambda x: fractions[x])

sample_df[['Tumor_Sample_Barcode', 'purity', 'subclonal.ix', 'fraction']].to_csv(cwd / 'fractions.tsv', sep='\t', index=False)