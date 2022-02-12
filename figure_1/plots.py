import numpy as np
import pandas as pd
import pylab as plt
file_path = 'files/'
import pickle
from scipy.stats import binom
from sklearn.cluster import KMeans

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'depth_norm':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('depth_norm')]
    import sys
    sys.path.append(str(cwd))

##TCGA data used
sample_df = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))

##https://gdc.cancer.gov/about-data/publications/pancanatlas
tcga_master_calls = pd.read_csv(cwd / 'files' / 'TCGA_mastercalls.abs_tables_JSedit.fixed.txt', sep='\t')
tcga_master_calls['sample_id'] = tcga_master_calls['sample'].apply(lambda x: x[:16])
tcga_master_calls = tcga_master_calls.loc[tcga_master_calls['call status'] == 'called']
tcga_master_calls = tcga_master_calls.groupby('sample_id')['purity'].mean().to_frame().reset_index()

sample_df = pd.merge(sample_df, tcga_master_calls, left_on='bcr_sample_barcode', right_on='sample_id', how='inner')

tcga_maf = pickle.load(open(cwd / 'files' / 'tcga_public_maf.pkl', 'rb'))
tcga_maf['vaf'] = tcga_maf['t_alt_count'].values / (tcga_maf['t_alt_count'].values + tcga_maf['t_ref_count'].values)
tcga_maf['depth'] = tcga_maf['t_ref_count'] + tcga_maf['t_alt_count']
tcga_maf['sample_id'] = tcga_maf['Tumor_Sample_Barcode'].apply(lambda x: x[:16])

cols = ['sample', 'subclonal.ix']
##https://gdc.cancer.gov/about-data/publications/pancan-aneuploidy
clonality_maf = pd.read_csv(cwd / 'files' / 'TCGA_consolidated.abs_mafs_truncated.fixed.txt', sep='\t', usecols=cols, low_memory=False)
result = clonality_maf.groupby('sample')['subclonal.ix'].apply(lambda x: sum(x) / len(x)).to_frame().reset_index()
del clonality_maf

sample_df = pd.merge(sample_df, result, left_on='Tumor_Sample_Barcode', right_on='sample', how='inner')
clusters = 8

X = np.stack([sample_df.loc[sample_df['purity'] > .37]['purity'].values, sample_df.loc[sample_df['purity'] > .37]['subclonal.ix'].values], axis=-1)
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
mapping = {i: j for i, j in zip(sample_df.loc[sample_df['purity'] > .37]['Tumor_Sample_Barcode'].values, kmeans.labels_)}

sample_df['bin'] = sample_df['Tumor_Sample_Barcode'].apply(lambda x: mapping.get(x, -1))
tcga_maf = pd.merge(tcga_maf, sample_df[['Tumor_Sample_Barcode', 'bin']], on='Tumor_Sample_Barcode', how='inner')

##kmeans plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = np.array(prop_cycle.by_key()['color'])
colors = colors[[0, 7, 2, 6, 1, 13, -1, 14, 4]]
ax.scatter(sample_df['purity'], sample_df['subclonal.ix'], c=colors[sample_df['bin'].values], edgecolors='none', marker='.')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Percent Tumor', fontsize=16)
ax.set_yticks([i/10 for i in range(0, 11, 2)])
ax.set_yticklabels([i*10 for i in range(0, 11, 2)])
ax.set_ylabel('Percent Subclonal', fontsize=16)
ax.set_ylim(-.01, 1.01)
ax.spines['left'].set_bounds(0, 1)
ax.set_xticks([i/10 for i in range(1, 11)])
ax.set_xticklabels([str(i*10) for i in range(1, 11)])
ax.set_xlim(.1, 1.01)
ax.spines['bottom'].set_bounds(.1, 1)
plt.savefig(cwd / 'figure_1' / 'plots' / 'kmeans_plot.png', dpi=600)

with open(cwd / 'figure_1' / 'plots' / 'kmeans_data.pkl', 'wb') as f:
    pickle.dump([sample_df, colors
                 ], f)


##true VAF plots
for bin in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    if bin == 8:
        bin_true = tcga_maf.loc[(tcga_maf['bin'] == -1) & (tcga_maf['depth'] > 250)]['vaf'].values
    else:
        bin_true = tcga_maf.loc[(tcga_maf['bin'] == bin) & (tcga_maf['depth'] > 250)]['vaf'].values
    true_distribution = {i: 0 for i in np.round_(np.arange(0, 1.01, .01), 2)}
    true_distribution.update({i: j for i, j in zip(*np.unique(np.round_(bin_true, 2), return_counts=True))})
    true_distribution[0.00], true_distribution[1.00] = true_distribution[0.00] * 2, true_distribution[1.00] * 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(sorted(list(true_distribution.keys())),
           np.array(list(true_distribution.values()))[np.argsort(list(true_distribution.keys()))] / sum(true_distribution.values()),
           width=np.array([.005] + [.01] * 99 + [.005]) / 1.5, color=colors[bin])
    ax.set_yticks([])
    ax.set_xlim(0, .505)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, .5)
    ax.set_xlabel('VAF', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    plt.savefig(cwd / 'figure_1' / 'plots' / (str(bin) + '_bin.png'), dpi=600)
    with open(cwd / 'figure_1' / 'plots' / (str(bin) + '_bin_data.pkl'), 'wb') as f:
        pickle.dump(true_distribution, f)


##expected distribution
read_depths = [25, 50, 100]
for read_depth in read_depths:
    for bin in [0, 6, -1]:
        bin_true = tcga_maf.loc[(tcga_maf['bin'] == bin) & (tcga_maf['depth'] > 250)]['vaf'].values
        true_distribution = {i: 0 for i in np.round_(np.arange(0, 1.01, .01), 2)}
        true_distribution.update({i: j for i, j in zip(*np.unique(np.round_(bin_true, 2), return_counts=True))})
        true_distribution[0.00], true_distribution[1.00] = true_distribution[0.00] * 2, true_distribution[1.00] * 2
        distribution_weights = np.array(list(true_distribution.values())) / sum(true_distribution.values())
        x = np.arange(0, read_depth + 1)
        distributions = np.array([binom.pmf(x, read_depth, p) for p in true_distribution])
        weighted = np.sum(distribution_weights[:, np.newaxis] * distributions, axis=0)
        expected_data = {}
        for index, i in enumerate(np.round_(x / read_depth, 2)):
            expected_data[i] = expected_data.get(i, []) + [weighted[index]]
        expected_data = {i: np.mean(expected_data[i]) for i in expected_data}
        assert list(expected_data.keys()) == sorted(expected_data.keys())
        vaf_bins = np.array(list(expected_data.keys()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_yticks([])
        ax.set_xlim(-1 / len(vaf_bins) / 2, .5 + (1 / len(vaf_bins) / 2))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_bounds(0, .5)
        ax.set_xlabel('VAF', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.bar(vaf_bins[0], np.array(list(expected_data.values()))[0], width=1 / 1.5 / len(vaf_bins), color='k')
        ax.bar(vaf_bins[1:, ], np.array(list(expected_data.values()))[1:],  width=1 / 1.5 / len(vaf_bins), color=colors[bin])
        plt.savefig(cwd / 'figure_1' / 'plots' / (str(bin) + '_bin_expected_' + str(read_depth) + '.png'), dpi=600)
        with open(cwd / 'figure_1' / 'plots' / (str(bin) + '_expected_' + str(read_depth) + '_data.pkl'), 'wb') as f:
            pickle.dump(expected_data, f)


##observed
for read_depth in read_depths:
    for bin in [0, 6, -1]:
        bin_observed = tcga_maf.loc[(tcga_maf['bin'] == bin) & (tcga_maf['depth'] == read_depth)]['vaf'].values
        vaf_bins = np.round_(np.arange(0, read_depth + 1) / read_depth, 2)
        observed_data = {i: 0 for i in vaf_bins}
        observed_data.update({i: j for i, j in zip(*np.unique(np.round_(bin_observed, 2), return_counts=True))})
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_yticks([])
        ax.set_xlim(-1 / len(vaf_bins) / 2, .5 + (1 / len(vaf_bins) / 2))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_bounds(0, .5)
        ax.set_xlabel('VAF', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.bar(vaf_bins, np.array(list(observed_data.values())),  width=1 / 1.5 / len(vaf_bins), color=colors[bin])
        plt.savefig(cwd / 'figure_1' / 'plots' / (str(bin) + '_bin_observed_' + str(read_depth) + '.png'), dpi=600)
        with open(cwd / 'figure_1' / 'plots' / (str(bin) + '_observed_' + str(read_depth) + '_data.pkl'), 'wb') as f:
            pickle.dump(expected_data, f)


##filtering probabilities
filter_dict, sample_df = pickle.load(open(cwd / 'read_depth_normalization' / 'results' / 'probabilities.pkl', 'rb'))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.arange(8, 250), filter_dict[6].values(), s=5, label='Pure, clonal', color=colors[6])
ax.scatter(np.arange(8, 250), filter_dict[0].values(), s=5, label='Pure, subclonal', color=colors[0])
ax.scatter(np.arange(8, 250), filter_dict[-1].values(), s=5, label='Mixed', color=colors[-1])
ax.set_xlim(0, 250)
ax.set_ylim(0, .705)
ax.set_xlabel('Read Depth', fontsize=16)
ax.set_ylabel('Filtering Probability', fontsize=16)
ax.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_bounds(0, 250)
ax.spines['left'].set_bounds(0, .7)
plt.savefig(cwd / 'figure_1' / 'plots' / 'probabilities.png', dpi=600)
with open(cwd / 'figure_1' / 'plots' / 'filtering_probabilities_data.pkl', 'wb') as f:
    pickle.dump([list(filter_dict[6].values()), list(filter_dict[0].values()), list(filter_dict[-1].values())], f)
# plt.show()

##sample read depth distribution

depths = pickle.load(open(cwd / 'read_depth_distribution' / 'results' / 'depths.pkl', 'rb'))

tumor = list(depths.keys())[0]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between(depths[tumor][0], depths[tumor][1], alpha=0.5)
ax.set_yticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_bounds(0, 250)
ax.set_xlabel('Read Depth', fontsize=16)
ax.set_ylabel('Density', fontsize=16)
# plt.show()
plt.savefig(cwd / 'figure_1' / 'plots' / 'read_depths.png', dpi=600)

with open(cwd / 'figure_1' / 'plots' / 'read_depths.pkl', 'wb') as f:
    pickle.dump([depths[tumor][0], depths[tumor][1]], f)

##sample count distribution

normed = []
bin = sample_df.loc[sample_df['Tumor_Sample_Barcode'] == tumor]['bin'].values[0]
for i in range(10000):
    count = 0
    total = 0
    while count < 100:
        total += 1
        if np.random.random() > depths[tumor][2]:
            depth = np.random.choice(depths[tumor][0], size=1, p=depths[tumor][1] / sum(depths[tumor][1]))
            if depth >= 8:
                if np.random.random() > filter_dict[bin][int(depth)]:
                    count += 1
        else:
            count += 1
    normed.append(total)

counts = {i:0 for i in range(min(normed), max(normed) + 1)}
counts.update({i: j for i, j in zip(*np.unique(normed, return_counts=True))})

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(list(counts.keys()), np.array(list(counts.values())) / sum(counts.values()), alpha=.3, width=1)
ax.set_yticks([])
ax.set_xlim(0, 200)
ax.set_ylim(0., .12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_ylabel('Density', fontsize=16)
ax.set_xlabel('Estimated Mutation Counts', fontsize=16)
plt.savefig(cwd / 'figure_1' / 'plots' / 'counts_observed.png', dpi=600)
with open(cwd / 'figure_1' / 'plots' / 'counts_observed_data.pkl', 'wb') as f:
    pickle.dump([list(counts.keys()), np.array(list(counts.values())) / sum(counts.values())], f)
