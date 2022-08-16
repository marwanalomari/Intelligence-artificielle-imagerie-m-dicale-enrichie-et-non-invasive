# Marwan AL OMARI, Master 2 Enginerring of connected objects
# University of Poitiers, XLIM laboratory
# marwanalomari@yahoo.com
# Internship subject: Artificial intelligence for enhanced non-invasive medical imaging
# Internship location: University of Poitiers, XLIM and I3M laboratories (CNRS 7252)

# Import libraries
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.utils import data
from dataset_brain import Dataset_gan_F
from loss import dice_loss,dice_score
from model import netD, define_G, Unet
from utils import label2onehot,classification_loss,gradient_penalty,seed_torch,update_lr
import pickle
import pandas as pd

# Main program: Loading data, and training and testing the architecture U-Net
def main():
    file_path='./npy_train/train_T18T28F.npy'
    train_data=Dataset_gan_F(file_path)
    batch_size=1
    train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,drop_last=False,num_workers=7)
    unet=Unet()
    LEARNING_RATE=0.0001
    optimizer=torch.optim.Adam(unet.parameters(),lr=LEARNING_RATE)
    unet.cuda()
    unet.train()
    EPOCH=2
    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'batch_score'])

    model_save='./unet_miso_t.pth'

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
            
        for i,(t1ce,t18t2f,label) in enumerate(train_loader):
            info_c_ =torch.randint(3,(t18t2f.size(0),))
            info_c = label2onehot(info_c_,3).cuda()
            img=torch.zeros(t18t2f.size(0),t18t2f.size(1),t18t2f.size(2),t18t2f.size(3))
            for i,l in enumerate(info_c_):
                if l==0:
                    img[i]=t1ce[i]
                else:
                    break

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
        log.to_csv('./logunet_miso_t.csv', index=False)
            
        print('EPOCH %d: train_score = %.4f'%(epoch+1,batch_score))

    torch.save(unet.state_dict(), model_save)


# Algorithm launch
if __name__ == '__main__':
    main()

# End of the program