import nibabel as nib
import numpy as np
import os
from multiprocessing import Pool
import sys

# modality = ['t1', 't1ce', 't2', 'flair']


def normalize(dir, target_path, file):
    print(file)
    patient = nib.load(os.path.join(dir, file))
    img = patient.get_fdata()
    mean = img[img > 0].mean()
    std = img[img > 0].std()
    new_img = (img - mean) / std
    new_data = nib.Nifti1Image(new_img, patient.affine, patient.header)
    nib.save(new_data, os.path.join(target_path, file))

    return


if __name__ == '__main__':
    source_dir = sys.argv[1]
    temp_dir = '%s/test_data_cp' % sys.argv[2]
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    target_dir = '%s/test_data_norm' % sys.argv[2]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    prefix = 'Radiology' if 'Radiology' in os.listdir(source_dir) \
        else 'radiology'
    os.system('cp %s/%s/*/*.nii.gz %s' % (source_dir, prefix, temp_dir))
    # target_path = '../source_data/test_data_norm'

    files = os.listdir(temp_dir)

    p = Pool()
    for file in files:
        p.apply_async(normalize, args=(temp_dir, target_dir, file))

    p.close()
    p.join()

    os.system('rm -rf %s/test_data_cp' % sys.argv[2])