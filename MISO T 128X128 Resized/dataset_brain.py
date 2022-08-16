# Marwan AL OMARI, Master 2 Engineering of connected objects
# University of Poitiers
# marwanalomari@yahoo.com
# Internship subject: Artificial intelligence for enhanced non-invasive medical imaging
# Internship location: University of Poitiers, XLIM (CNRS 7252) and I3M (CHU Poitiers) laboratories 

'''Please cite the work code in case of use, thanks!'''

# Import libraries
from torch.utils import data
import numpy as np

class Dataset_gan_F(data.Dataset):
    def __init__(self,file):
        self.file_t18t2f=np.load(file)
        file_seg=file.replace('t18t2f','seg')
        file_t1ce=file.replace('t18t2f','t1ce')
        self.label=np.load(file_seg)
        self.file_t1ce=np.load(file_t1ce)
    def __getitem__(self, index):
        t18t2f=self.file_t18t2f[index][np.newaxis,:]
        t1ce=self.file_t1ce[index][np.newaxis,:]
        t18t2f = (t18t2f - 0.5) / 0.5
        t1ce = (t1ce - 0.5) / 0.5
        label=self.label[index][np.newaxis,:]
        return t1ce,t18t2f,label

    def __len__(self):
        return int(len(self.file_t18t2f))