# Marwan AL OMARI, Master 2 Engineering of connected objects
# University of Poitiers
# marwanalomari@yahoo.com
# Internship subject: Artificial intelligence for enhanced non-invasive medical imaging
# Internship location: University of Poitiers, XLIM (CNRS 7252) and I3M (CHU Poitiers) laboratories 

'''Please cite the work code in case of use, thanks!'''

# Import libraries
from torch.utils import data
import numpy as np

class Dataset_gan_T(data.Dataset):
    def __init__(self,file):
        self.file_t18t2=np.load(file)
        file_seg=file.replace('t18t2','seg')
        file_flair=file.replace('t18t2','flair')
        #file_t1ce=file.replace('t1','t1ce')
        #file_t2=file.replace('t1','t2')
        self.label=np.load(file_seg)
        self.file_flair=np.load(file_flair)
        #self.file_t1ce=np.load(file_t1ce)
        #self.file_t2=np.load(file_t2)
    def __getitem__(self, index):
        flair=self.file_flair[index][np.newaxis,:]
        t18t2=self.file_t18t2[index][np.newaxis,:]
        #t1ce=self.file_t1ce[index][np.newaxis,:]
        #t2=self.file_t2[index][np.newaxis,:]
        flair=(flair-0.5)/0.5
        t18t2 = (t18t2 - 0.5) / 0.5
        #t1ce = (t1ce - 0.5) / 0.5
        #t2 = (t2 - 0.5) / 0.5
        label=self.label[index][np.newaxis,:]
        return flair,t18t2,label

    def __len__(self):
        return int(len(self.file_t18t2))