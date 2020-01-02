#!/bin/sh

cd /submission

echo "Get patient id, saved to /data/output"
python get_id.py /data/images /data/output

echo "Pathology data processing and save patches to /data/output. This process\
will take several hours depends on data set size"
python pathology_processing.py /data/images /data/output

echo "Pre-processing for MRI data, will save to /data/output"
python norm.py /data/images /data/output

echo "Predict tumor segmentation mask"
python single_validation.py --gpu 0 --data 2 --norm all --batch-size 1 \
--basefilter 8 --seed 4096 --lr 0.01 --epoch 240 --fp16 --loss brats19v2 \
--schedule s1 --net v15 --resume --comment single_train_v15_240 --outdir /data/output

echo "Predict tumor class, save classification.csv to /data/output"
python predict_newmodel.py --gpu 0 --data 2 --net newmodel --comment new_8 \
--batch-size 1 --lr 0.01 --seed 24 --fp16 --resume --outdir /data/output