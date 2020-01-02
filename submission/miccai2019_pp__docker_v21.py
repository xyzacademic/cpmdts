# -*- coding: utf-8 -*-
"""
Created on Tue Sep 3 23:58:00 2019

@author: ffarhat
"""
#%%
import openslide
from io import BytesIO
import os
import sys

#%%
home_dir = "/DeepLearning/CPM-2019/miccai2019/CPM-RadPath_2019_"

if len(sys.argv) == 1:
    #slide_type = "Training"; slide_folder = "class_a"; training_class = "A"; output_folder = "patches.a"
    slide_type = "Testing"; slide_folder = "class_t"; training_class = "T"; output_folder = "patches.t"
else:
    slide_type = str(sys.argv[1])
    slide_folder = str(sys.argv[2])
    training_class = str(sys.argv[3]); #snippet that uses this class is removed from this Docker version of script
    output_folder = str(sys.argv[4])

#pstep = patch step to skip patches; sdf = scalde down factor
k = 0; pstep = 1; patchsize = 512; sdf = 1; downsize = int(patchsize/sdf);

for root, dirs, files in os.walk(home_dir+slide_type+"_Data/Pathology/"+slide_folder): #walk through
    for file in files:
        if file.endswith(".tiff"): #in native folders
            #print("WSI Counter = ", k)
            k = k + 1; #print(file)
            fullpath = os.path.join(root, file); #print(fullpath)
            fullpathf = fullpath.replace('\\', '/'); #print(fullpathf)
            patient_id = file[0:-5]; #print("Patient ID:", patient_id)
            biopsy_img = openslide.OpenSlide(fullpathf)
            
            biopsy_ldim = biopsy_img.level_dimensions[0];

            xstep = int(biopsy_ldim[0]/patchsize); ystep = int(biopsy_ldim[1]/patchsize); file_size_dict = {}

            top_patches = int(0.0016*xstep*ystep*3); counter = 0;
            
            if (top_patches < 36):
                top_patches = 36

            for i in range (0, xstep, pstep):
                for j in range (0, ystep, pstep):
                    biopsy_img0 = biopsy_img.read_region((i*patchsize,j*patchsize), 0, (patchsize,patchsize))
                    
                    mem_file = BytesIO(); biopsy_img0.save(mem_file, 'png'); mem_file_size = mem_file.tell()
                    
                    file_size_dict[(i*patchsize,j*patchsize)] = mem_file_size
                   
                    #print("Set:", slide_folder, "| Slide:", k, "| Patch:", counter, "| Limits: [", xstep, ":", ystep, "] | Location: [", i, ":", j, "] | File Size: ", mem_file_size)

                    counter = counter + 1
            
            sorted_fsd = sorted(file_size_dict.items(), key=lambda kv: kv[1], reverse = True)
            
            print("Top Patches:", top_patches)
            
            for i in range(min(top_patches,len(sorted_fsd))):
                biopsy_img0 = biopsy_img.read_region(sorted_fsd[i][0], 0, (patchsize,patchsize))
                #print("Save sub-image ["+str(sorted_fsd[i][0])+"] to disk as PNG file:")
                biopsy_img0.save(home_dir+slide_type+"_Data/Pathology/"+output_folder+"/"+patient_id+"_"+str(sorted_fsd[i][0][0])+"_"+str(sorted_fsd[i][0][1])+".png", "PNG")
            
#%%
