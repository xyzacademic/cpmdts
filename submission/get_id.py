#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pandas as pd
import os
import sys


if __name__ == '__main__':
    source_dir = sys.argv[1]
    # temp_dir = 'temp'
    temp_dir = sys.argv[2]
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    prefix = 'Pathology' if 'Pathology' in os.listdir(source_dir) \
        else 'pathology'

    patient_id = [os.path.splitext(file)[0]
                  for file in os.listdir(os.path.join(source_dir, prefix))]
    patient_id.sort()
    df = pd.DataFrame({'CPM_RadPath_2019_ID': patient_id})
    df.to_csv(os.path.join(temp_dir, 'patient_id.csv'), index=False)

