##download from https://www.synapse.org/#!Synapse:syn21785741
from tqdm import tqdm
import subprocess
file_path = '/home/janaya2/Desktop/broad_wigs/wigs/'
cmd = ['ls', file_path]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
files = [str(i, 'utf-8') for i in p.communicate()[0].split() if '.gz' in str(i)]
files = sorted(files)

for file in tqdm(files):
    name = file.split('.gz')[0] + '.bed'
    cmd = "zcat " + file_path + file + " | wig2bed -d | awk '$5 == 1' | bedtools merge > " + file_path + '/beds/' + name
    subprocess.run(cmd, shell=True)
