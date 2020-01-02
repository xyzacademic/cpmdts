# build-in library
import os
from time import time

# third-party library
import numpy as np
import torch
from torch.backends import cudnn
from torch import optim
import torch.nn.functional as F
from torch import nn
import torch.utils.data
import pandas as pd
import apex
from apex import amp
import nibabel as nib

# from easylab.dl.log import record_csv
# from easylab.dl.utils import AverageMeter
# from easylab.dl.utils import evaluation
# from easylab.dl.model.model import Model
# from easylab.dl.model.model import reduce_tensor
from model import Model


class Brats19Segmenter(Model):


    def inference(self, test_loader=None, prob=False, target_path=None, replace_origin='anat', replace_target='pred'):
        config = self.config
        assert self.net is not None
        root = test_loader.dataset.root
        self.net.eval()
        start = time()

        if not os.path.exists(target_path):
            os.makedirs(target_path)
        with torch.no_grad():
            for batch_idx, (data, filenames) in enumerate(test_loader):
                if config['use_cuda'] is True:
                    data = data.cuda()

                outputs, wt, tc, et, ed, ncr = self.net(data)
                if prob is True:
                    pre = F.softmax(outputs, dim=1)[:, 1]
                else:
                    pre = outputs.max(1)[1].data.cpu().numpy()

                for i in range(pre.shape[0]):
                    patient = nib.load(os.path.join(root, filenames[i]))
                    new_pre = np.zeros(shape=(data.size(0), 240, 240, 155), dtype=np.uint8)
                    new_pre[:, 38:198, 28:220, :] = pre[:, :, :, 5:]
                    new_data = nib.Nifti1Image(new_pre[i], patient.affine, patient.header)
                    nib.save(new_data, os.path.join(target_path, filenames[i].replace(replace_origin, replace_target)))
                    print('Save {file} successfully.'.format(file=os.path.join(target_path,
                                                        filenames[i].replace(replace_origin, replace_target))))
        print('This inference cost %.2f seconds' % (time() - start))
