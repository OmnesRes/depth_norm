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
mc3_pcawg = pd.read_csv(cwd / 'files' / 'mc3_pcawg.csv', sep=',') ##from https://mbailey.shinyapps.io/MAFit/
mc3_pcawg = mc3_pcawg.loc[mc3_pcawg['id_match'] > 5]
sample_df = sample_df.loc[sample_df['Tumor_Sample_Barcode'].isin(mc3_pcawg['mc3_exome_barcode'])]
with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
    for sample, result in tqdm(zip(sample_df['Tumor_Sample_Barcode'].values, executor.map(get_fraction, sample_df['Tumor_Sample_Barcode'].values))):
        fractions[sample] = result

sample_df['fraction'] = sample_df['Tumor_Sample_Barcode'].apply(lambda x: fractions[x])
temp_df = pd.merge(sample_df, mc3_pcawg, left_on='Tumor_Sample_Barcode', right_on='mc3_exome_barcode', how='inner')

pearsonr(temp_df['fraction'].values, (temp_df['id_mc3'].values + temp_df['id_match'].values) / (temp_df['id_mc3'].values + temp_df['id_match'].values + temp_df['id_pcawg'].values))
#(0.450482862209613, 1.0344248965249786e-32)
pearsonr(temp_df['purity'].values, (temp_df['id_mc3'].values + temp_df['id_match'].values) / (temp_df['id_mc3'].values + temp_df['id_match'].values + temp_df['id_pcawg'].values))
#(0.2325480512371445, 3.7003654597729775e-09)

fig = plt.figure()
fig.subplots_adjust(
top=0.994,
bottom=0.114,
left=0.116,
right=0.992,
hspace=0.078,
wspace=0.0)
gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
sns.regplot(temp_df['fraction'].values,
            (temp_df['id_mc3'].values + temp_df['id_match'].values) / (temp_df['id_mc3'].values + temp_df['id_match'].values + temp_df['id_pcawg'].values),
            truncate=False,
            ax=ax1,
            scatter_kws={'s': 10,
                         'edgecolors': 'none',
                         'marker': '.',
                         'alpha':.5},
            line_kws={'lw': 1},
              )
counts = {round(i, 2): 0 for i in np.arange(min(np.round(temp_df['fraction'].values, 2)), max(np.round(temp_df['fraction'].values, 2)) + .01, .01)}
counts.update(temp_df['fraction'].apply(lambda x: round(x, 2)).value_counts().to_dict())
ax2.bar(list(counts.keys()), np.array(list(counts.values())) / sum(counts.values()), width=.01, alpha=.3)
x_min = .69
x_max = .96
y_min = .64
y_max = 1.01
ax1.set_xlim(.695, .955)
ax2.set_xlim(.695, .955)
ax1.set_ylim(y_min, y_max)
ax1.set_xticks([])
ax2.set_yticks([])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax1.spines['left'].set_bounds(.65, 1)
ax2.spines['bottom'].set_bounds(.7, .95)
ax2.set_xlabel('Estimated MC3 Fraction Observed', fontsize=16)
ax1.set_ylabel('MC3 / (MC3 $\cup$ PCAWG)', fontsize=16)
plt.savefig(cwd / 'figure_2' / 'pcawg_plot.png', dpi=600)
