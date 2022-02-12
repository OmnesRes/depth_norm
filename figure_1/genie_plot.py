import numpy as np
import pandas as pd
import pylab as plt
file_path = 'files/'
import pickle

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'depth_norm':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('depth_norm')]
    import sys
    sys.path.append(str(cwd))


#https://www.synapse.org/#!Synapse:syn25895958
usecols = ['t_depth', 'Center', 'Tumor_Sample_Barcode']
genie = pd.read_csv(cwd / 'files' / 'genie' / 'data_mutations_extended.txt', delimiter='\t',
                    usecols=usecols,
                    )

samples = pd.read_csv(cwd / 'files' / 'genie' / 'data_clinical_sample.txt', delimiter='\t', skiprows=4)

genie = pd.merge(genie, samples[['SAMPLE_ID', 'SEQ_ASSAY_ID']], left_on='Tumor_Sample_Barcode', right_on='SAMPLE_ID')

msk = genie.loc[(genie['SEQ_ASSAY_ID'] == 'MSK-IMPACT468') & (genie['t_depth'] < 1500)]
dfci = genie.loc[(genie['SEQ_ASSAY_ID'].str.contains('DFCI-ONCOPANEL-3')) & (genie['t_depth'] < 1500)]
vicc = genie.loc[(genie['SEQ_ASSAY_ID'] == 'VICC-01-T7') & (genie['t_depth'] < 1500)]

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
tcga_maf = pd.merge(tcga_maf, sample_df[['Tumor_Sample_Barcode']], on='Tumor_Sample_Barcode', how='inner')
tcga_maf = tcga_maf.loc[tcga_maf['depth'] < 1500]

fig = plt.figure()
ax = fig.add_subplot(111)
counts = tcga_maf['depth'].value_counts().to_dict()
bin = 5
values = [sum([counts.get(i + j, 0) for j in range(bin)]) for i in range(min(counts.keys()), max(counts.keys()) + 1, bin)]
ax.bar([i for i in range(min(counts.keys()), max(counts.keys()) + 1, bin)], np.array(values) / sum(values) / bin, width=bin, alpha=.3, label='MC3 Public')
counts = msk['t_depth'].value_counts().to_dict()
bin = 10
values = [sum([counts.get(i + j, 0) for j in range(bin)]) for i in range(int(min(counts.keys())), int(max(counts.keys()) + 1), bin)]
ax.bar([i for i in range(int(min(counts.keys())), int(max(counts.keys()) + 1), bin)], np.array(values) / sum(values) / bin, width=bin, alpha=.3, label='GENIE MSK-IMPACT468')
counts = dfci['t_depth'].value_counts().to_dict()
values = [sum([counts.get(i + j, 0) for j in range(bin)]) for i in range(int(min(counts.keys())), int(max(counts.keys()) + 1), bin)]
ax.bar([i for i in range(int(min(counts.keys())), int(max(counts.keys()) + 1), bin)], np.array(values) / sum(values) / bin, width=bin, alpha=.3, label='GENIE DFCI-ONCOPANEL-3')
counts = vicc['t_depth'].value_counts().to_dict()
values = [sum([counts.get(i + j, 0) for j in range(bin)]) for i in range(int(min(counts.keys())), int(max(counts.keys()) + 1), bin)]
ax.bar([i for i in range(int(min(counts.keys())), int(max(counts.keys()) + 1), bin)], np.array(values) / sum(values) / bin, width=bin, alpha=.3, label='GENIE VICC-01-T7')
ax.set_yticks([])
ax.set_xticks([0, 300, 600, 900, 1200, 1500])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_bounds(0, 1500)
ax.set_xlabel('Read Depth', fontsize=16)
ax.set_ylabel('Density', labelpad=-15, fontsize=16)
ax.legend(frameon=False, ncol=2, loc=(.1, .85))
plt.savefig(cwd / 'figure_1' / 'plots' / 'genie_plot.png', dpi=600)

with open(cwd / 'figure_1' / 'plots' / 'gene_data.pkl', 'wb') as f:
    pickle.dump([tcga_maf['depth'].value_counts().to_dict(),
                 msk['t_depth'].value_counts().to_dict(),
                 dfci['t_depth'].value_counts().to_dict(),
                 vicc['t_depth'].value_counts().to_dict()
                 ], f)