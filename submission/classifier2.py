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

from model import Model



class Classifier(Model):


    def inference(self, epoch=0, test_loader=None, prob=False, source_file='1.csv', target_path=None,
                  replace_origin='anat', replace_target='pred'):
        config = self.config
        assert self.net is not None
        root = test_loader.dataset.root
        self.net.eval()
        start = time()

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        df = pd.read_csv(os.path.join(target_path, source_file))
        # df.drop(['age_in_days'], axis=1, inplace=True)
        df['class'] = '1'
        results = []
        with torch.no_grad():
            for batch_idx, (data1, data2, filenames) in enumerate(test_loader):
                data = [data1, data2]
                if config['use_cuda'] is True:
                    if isinstance(data, list):
                        for i in range(len(data)):
                            data[i] = data[i].cuda()
                    else:
                        data = data.cuda()

                alpha = 1
                outputs = self.net(data)

                if prob is True:
                    pre = F.softmax(outputs, dim=1).data.cpu().numpy()
                else:
                    pre = outputs.max(1)[1].data.cpu().numpy()

                for i in range(pre.shape[0]):
                    if pre[i] == 0:
                        target = 'A'
                    elif pre[i] == 2:
                        target = 'O'
                    elif pre[i] == 1:
                        target = 'G'

                    df.loc[df['CPM_RadPath_2019_ID']==filenames[i].replace(replace_origin, replace_target), 'class'] = target
        df.to_csv(os.path.join(target_path, 'output_classification.csv'), index=False)

                # results.append(pre)
        #     results = np.concatenate(results, axis=0)
        #     np.save('test_pred_mask_prob.npy', results)
        print('This inference cost %.2f seconds' % (time() - start))

