# Marwan AL OMARI, Master 2 Engineering of connected objects
# University of Poitiers
# marwanalomari@yahoo.com
# Internship subject: Artificial intelligence for enhanced non-invasive medical imaging
# Internship location: University of Poitiers, XLIM (CNRS 7252) and I3M (CHU Poitiers) laboratories 

'''Please cite the work code in case of use, thanks!'''

# Import libraries
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.utils import data
from dataset_brain import Dataset_gan
from loss import dice_loss,dice_score
from model import netD, define_G, Unet
from utils import label2onehot,classification_loss,gradient_penalty,seed_torch,update_lr
import pickle
import pandas as pd

# Main program: Loading data, and training and testing the architecture U-Net
def main():
    seed_torch(10)
    file_path='./Pdataset/FULL/npy_train/train_t1.npy'
    model_save='./Weights/unet_simo_cls.pth'
    train_data=Dataset_gan(file_path)
    batch_size=1
    train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,drop_last=False,num_workers=7)
    unet=Unet()
    LEARNING_RATE=0.0001
    optimizer=torch.optim.Adam(unet.parameters(),lr=LEARNING_RATE)
    unet.cuda()
    unet.train()
    EPOCH=100
    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'batch_score'])
    for epoch in range(EPOCH):
        batch_score=0
        num_batch=0
        if epoch==25:
            update_lr(optimizer,0.0002)
            print('*****Change lr to :',optimizer.param_groups[0]['lr'])
        elif epoch==50:
            update_lr(optimizer,0.00005)
            print('*****Change lr to :',optimizer.param_groups[0]['lr'])
        elif epoch==75:
            update_lr(optimizer,0.00003)
            print('***** Change lr to :',optimizer.param_groups[0]['lr'])
        elif epoch==100:
            update_lr(optimizer,0.00002)
            print('***** Change lr to :',optimizer.param_groups[0]['lr'])
        elif epoch==125:
            update_lr(optimizer,0.00001)
            print('***** Change lr to :',optimizer.param_groups[0]['lr'])
        elif epoch==150:
            update_lr(optimizer,0.00004)
            print('***** Change lr to :',optimizer.param_groups[0]['lr'])
        elif epoch==175:
            update_lr(optimizer,0.00004)
            print('***** Change lr to :',optimizer.param_groups[0]['lr'])
        for i,(flair,t1,t1ce,t2,label) in enumerate(train_loader):
            info_c_ =torch.randint(3,(t1.size(0),))
            info_c = label2onehot(info_c_,3).cuda()
            img=torch.zeros(t1.size(0),t1.size(1),t1.size(2),t1.size(3))
            for i,l in enumerate(info_c_):
                if l==0:
                    img[i]=flair[i]
                elif l==1:
                    img[i]=t1ce[i]
                elif l==2:
                    img[i]=t1[i]
            seg=unet(img.float().cuda(),info_c)
            loss=dice_loss(seg,label.float().cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            seg=seg.cpu()
            seg[seg>=0.5]=1.
            seg[seg!=1]=0.
            batch_score+=dice_score(seg,label.float()).data.numpy()
            num_batch+=img.size(0)

        batch_score/=num_batch

        tmp = pd.Series([
            epoch,
            LEARNING_RATE,
            batch_score,
        ],index=['epoch', 'lr','batch_score'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('./Weights/logunet_simo_cls.csv', index=False)
            
        print('EPOCH %d: train_score = %.4f'%(epoch+1,batch_score))

    torch.save(unet.state_dict(), model_save)


# Algorithm launch
if __name__ == '__main__':
    main()

# End of the program