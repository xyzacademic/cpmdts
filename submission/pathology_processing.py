#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""Pathology processing

Patch pathology data (tiff)

Reference:
Link:

Updating log:
V1.0 description
"""

import openslide
from io import BytesIO
import os
import sys
import pandas as pd
from multiprocessing import Pool
import time


def save_patch(name, full_path, postfix, patch_step, patch_size, temp_dir):
    print(name)
    file = os.path.join(full_path, name + postfix)
    biopsy_img = openslide.OpenSlide(file)
    biopsy_ldim = biopsy_img.level_dimensions[0]

    xstep = biopsy_ldim[0] // patch_size
    ystep = biopsy_ldim[1] // patch_size
    file_size_dict = {}
    top_patches = int(0.0016 * xstep * ystep * 3)
    counter = 0

    if top_patches < 36:
        top_patches = 36

    for i in range(0, xstep, patch_step):
        for j in range(0, ystep, patch_step):
            biopsy_img0 = biopsy_img.read_region(
                (i * patch_size, j * patch_size), 0, (patch_size, patch_size)
            )
            mem_file = BytesIO()
            biopsy_img0.save(mem_file, 'png')
            mem_file_size = mem_file.tell()
            file_size_dict[(i * patch_size, j * patch_size)] = mem_file_size

            counter += 1
            del biopsy_img0, mem_file, mem_file_size

    sorted_fsd = sorted(file_size_dict.items(), key=lambda kv: kv[1], reverse=True)
    print('Top Patches: ', top_patches)

    for i in range(min(top_patches, len(sorted_fsd))):
        biopsy_img0 = biopsy_img.read_region(sorted_fsd[i][0], 0, (patch_size, patch_size))
        end_fix = '_' + str(sorted_fsd[i][0][0]) + '_' + str(sorted_fsd[i][0][1]) + '.png'
        biopsy_img0.save(os.path.join(temp_dir, name + '%s' % end_fix), 'PNG')
        del biopsy_img0

    return


if __name__ == '__main__':

    source_dir = sys.argv[1]
    temp_dir = '%s/all_patches/unknow' % sys.argv[2]
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    prefix = 'Pathology' if 'Pathology' in os.listdir(source_dir) \
        else 'pathology'
    postfix = '.tiff'
    full_path = os.path.join(source_dir, prefix)
    # Parameters
    k = 0
    patch_step = 1
    patch_size = 512
    sdf = 1
    downsize = patch_size // sdf

    patient_id = pd.read_csv(os.path.join(sys.argv[2], 'patient_id.csv'))['CPM_RadPath_2019_ID'].values

    pool = Pool()
    results = []
    start_time = time.time()
    for name in patient_id:
        results.append(
            pool.apply_async(save_patch,
            args=(name, full_path, postfix, patch_step, patch_size, temp_dir))
        )

    pool.close()
    pool.join()

    [result.get() for result in results]
    print('Finished in %.2f seconds' % (time.time() - start_time))