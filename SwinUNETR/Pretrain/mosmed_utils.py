import glob
import json

# read in the list of files
files = nii_files = list(glob.glob("/ocean/projects/asc170022p/lisun/copd/covid/dataset/ru/studies/CT-*/*.nii.gz"))

n_train = int(0.95 * len(nii_files))
n_validation = len(nii_files) - n_train

d = {}
d['training'] = []
for i in range(0, n_train):
    t = {}
    t['image'] = nii_files[i]
    d['training'].append(t)

d['validation'] = []
for i in range(n_train, len(nii_files)):
    v = {}
    v['image'] = nii_files[i]
    d['validation'].append(v)

# Save JSON file
with open('./jsons/dataset_MosMed_nii.json', 'w') as f:
    json.dump(d, f)