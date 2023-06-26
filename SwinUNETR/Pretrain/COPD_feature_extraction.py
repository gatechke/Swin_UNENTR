# Extract features from COPDGene dataset

import os
import torch

import numpy as np
from tqdm import tqdm

from monai.data import DataLoader, Dataset, load_decathlon_datalist
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AddChanneld,
    Compose,
    #CropForegroundd,
    LoadImaged,
    Orientationd,
    #RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    #SpatialPadd,
    ToTensord,
    Resized,
)

def extract_sid(filename):
    words = filename.split('/')[-1].split('_')
    # join words with '_
    return '_'.join(words[:5])

# load model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = ensure_tuple_rep(7, 3)
patch_size = ensure_tuple_rep(2, 3)

model = SwinViT(
    in_chans=1,
    embed_dim=48,
    window_size=window_size,
    patch_size=patch_size,
    depths=[2, 2, 2, 2],
    num_heads=[3, 6, 12, 24],
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=torch.nn.LayerNorm,
    spatial_dims=3).to(device)

exp_name = 'mdl_512_96_mul'
exp_dir = './runs/' + exp_name + '/'
ckp = torch.load(exp_dir + 'model_bestValRMSE.pt')
# remove prefix 'module.SwinViT.' from pretrained weights and only keep weights starting with 'module.swinViT.'
weight = {}
for k, v in ckp['state_dict'].items():
    if k.startswith('module.swinViT.'):
        weight[k.replace('module.swinViT.', '')] = v
model.load_state_dict(weight)
print("Using pretrained self-supervied Swin UNETR backbone weights !")

# load dataset
splits1 = "/dataset_COPDGene_nii.json"
list_dir = "./jsons"
datadir1 = ''
jsonlist1 = list_dir + splits1
trainlist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
vallist1 = load_decathlon_datalist(jsonlist1, False, "validation", base_dir=datadir1)
val_files = trainlist1 + vallist1

# load dependent variable
sid_arr = np.load('./sid_arr_full.npy')
dep_arr = np.load('./feature_arr_patch_full.npy')

val_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Resized(keys=["image"], spatial_size=[288, 288, 288]),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=240, b_min=-1.0, b_max=1.0, clip=True
        ),
        #SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
        #CropForegroundd(keys=["image"], source_key="image", k_divisible=[96, 96, 96]),
        ToTensord(keys=["image"]),
    ]
)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=5, shuffle=False, drop_last=True)

# define lists to store sid, image emb and dependent variable
sid_lst = []
emb_lst = []
dep_lst = []

model.eval()
with torch.no_grad():
    for step, batch in enumerate(tqdm(val_loader)):
        # extract sid from filename
        filename = batch['image_meta_dict']['filename_or_obj'][0]
        sid = extract_sid(filename)
        try:
            # find the index of sid in sid_arr
            idx = np.where(sid_arr == sid)[0][0]
            # extract dependent variable
            dep = dep_arr[idx]
            # get image embedding
            x = batch['image'].cuda()
            emb_patch = sliding_window_inference(x, (96, 96, 96), 4, model, overlap=0)
            emb_image = emb_patch[4].mean(dim=(0, 2, 3, 4))
            # append to lists
            sid_lst.append(sid)
            emb_lst.append(emb_image.cpu().numpy())
            dep_lst.append(dep)
        except:
            pass

# save sid, image emb and dependent variable
np.save(exp_dir + 'sid_arr_full.npy', np.array(sid_lst))
np.save(exp_dir + 'pred_arr_patch_full.npy', np.array(emb_lst))
np.save(exp_dir + 'feature_arr_patch_full.npy', np.array(dep_lst))
