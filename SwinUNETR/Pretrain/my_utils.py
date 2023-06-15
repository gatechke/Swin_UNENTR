import os
import glob
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from tqdm import tqdm
import json

def convert_mhd_to_nii(src_dir, dest_dir):
    # Find all .mhd files in src_dir
    mhd_files = glob.glob(os.path.join(src_dir, '*.mhd'))

    for mhd_file in tqdm(mhd_files):
        # Read the .mhd file
        mhd_image = sitk.ReadImage(mhd_file)

        # Convert the SimpleITK image to a numpy array
        numpy_image = sitk.GetArrayFromImage(mhd_image)

        # Create a nibabel Nifti1Image from the numpy array
        nii_image = nib.Nifti1Image(numpy_image, np.eye(4))

        # Create the destination file path
        base_name = os.path.basename(mhd_file)  # Get the base name of the .mhd file
        base_name_without_ext = os.path.splitext(base_name)[0]  # Remove the .mhd extension
        dest_file = os.path.join(dest_dir, base_name_without_ext + '.nii.gz')  # Create the .nii.gz file path

        # Save the Nifti1Image as a .nii.gz file
        nib.save(nii_image, dest_file)


# Usage:
#src_dir = '/ocean/projects/asc170022p/shared/Data/ChallengeDatasets/LUNA16/subset1'
#dest_dir = '/ocean/projects/asc170022p/yuke/Downloads/LUNA16'
#convert_mhd_to_nii(src_dir, dest_dir)


# Create a JSON file
#dest_dir = '/ocean/projects/asc170022p/yuke/Downloads/LUNA16'
#nii_files = glob.glob(os.path.join(dest_dir, '*.nii.gz'))

nii_files = list(glob.glob("/ocean/projects/asc170022p/shared/Data/COPDGene/Images/*/Phase-1/Isotropic/*_INSP_STD_*_COPD_BSpline_Iso1.0mm.nii.gz"))
# remove error files
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/23305X/Phase-1/Isotropic/23305X_INSP_STD_HAR_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/22713H/Phase-1/Isotropic/22713H_INSP_STD_TEM_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/11260L/Phase-1/Isotropic/11260L_INSP_STD_HAR_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/24826E/Phase-1/Isotropic/24826E_INSP_STD_NJC_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/22851T/Phase-1/Isotropic/22851T_INSP_STD_USD_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/19185L/Phase-1/Isotropic/19185L_INSP_STD_COL_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/17739S/Phase-1/Isotropic/17739S_INSP_STD_COL_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/24034X/Phase-1/Isotropic/24034X_INSP_STD_UAB_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/25026D/Phase-1/Isotropic/25026D_INSP_STD_BWH_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/22980E/Phase-1/Isotropic/22980E_INSP_STD_BAY_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/18910H/Phase-1/Isotropic/18910H_INSP_STD_UMC_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/16138N/Phase-1/Isotropic/16138N_INSP_STD_AVA_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/10071D/Phase-1/Isotropic/10071D_INSP_STD_NJC_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/19843X/Phase-1/Isotropic/19843X_INSP_STD_NJC_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/18043M/Phase-1/Isotropic/18043M_INSP_STD_JHU_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/11889H/Phase-1/Isotropic/11889H_INSP_STD_MSM_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/25272S/Phase-1/Isotropic/25272S_INSP_STD_MSM_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/25320D/Phase-1/Isotropic/25320D_INSP_STD_FAL_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/22481M/Phase-1/Isotropic/22481M_INSP_STD_USD_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/25759U/Phase-1/Isotropic/25759U_INSP_STD_UIA_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/26087C/Phase-1/Isotropic/26087C_INSP_STD_MSM_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/25357A/Phase-1/Isotropic/25357A_INSP_STD_UIA_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/11630S/Phase-1/Isotropic/11630S_INSP_STD_HAR_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/13207R/Phase-1/Isotropic/13207R_INSP_STD_UIA_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/24597P/Phase-1/Isotropic/24597P_INSP_STD_NJC_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/25403H/Phase-1/Isotropic/25403H_INSP_STD_NJC_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/24095R/Phase-1/Isotropic/24095R_INSP_STD_DUK_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/13485T/Phase-1/Isotropic/13485T_INSP_STD_HAR_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/22856D/Phase-1/Isotropic/22856D_INSP_STD_BWH_COPD_BSpline_Iso1.0mm.nii.gz')
nii_files.remove('/ocean/projects/asc170022p/shared/Data/COPDGene/Images/14457T/Phase-1/Isotropic/14457T_INSP_STD_NJC_COPD_BSpline_Iso1.0mm.nii.gz')

# this line is temp for testing
#nii_files = nii_files[7964:]

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
with open('./jsons/dataset_COPDGene_nii.json', 'w') as f:
    json.dump(d, f)
