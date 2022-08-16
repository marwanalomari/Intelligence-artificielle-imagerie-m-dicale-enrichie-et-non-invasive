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
from dataset_brain import Dataset_gan_F
from loss import dice_loss,dice_score
from model import netD, define_G, Unet
from utils import label2onehot,classification_loss,gradient_penalty,seed_torch,update_lr
import pickle
import pandas as pd


def main():
    batch_size=1
    test_path='E:/Marwan/MISOF/npy_test/test_T18T28F.npy'
    test_data=Dataset_gan_F(test_path)
    fix_loader=data.DataLoader(dataset=test_data,batch_size=batch_size,num_workers=7)
    fix_iter=iter(fix_loader)

    for i in range(1):
        next(fix_iter)
    t1ce_fix,t18t2f_fix,seg_fix=next(fix_iter)
    origin_fix = np.hstack((t18t2f_fix[0][0],t1ce_fix[0][0],seg_fix[0][0]))

    for i in range(1):
        next(fix_iter)
    t1ce_fix_2,t18t2f_fix_2,seg_fix_2=next(fix_iter)
    origin_fix_2 = np.hstack((t18t2f_fix_2[0][0],t1ce_fix_2[0][0],seg_fix_2[0][0]))

    for i in range(1):
        next(fix_iter)
    t1ce_fix_3,t18t2f_fix_3,seg_fix_3=next(fix_iter)
    origin_fix_3 = np.hstack((t18t2f_fix_3[0][0],t1ce_fix_3[0][0],seg_fix_3[0][0]))

    del fix_loader,fix_iter,test_data


    seed_torch()
    file_path='E:/Marwan/MISOF/npy_gan/gan_T18T28F.npy'
    train_data=Dataset_gan_F(file_path)

    train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=7)

    generator = define_G(4, 1, 1, 'unet_128', norm='instance', )
    discriminator=netD()
    unet = Unet()
    unet.load_state_dict(torch.load("E:/Marwan/MISOF/Weights/unet_miso_t.pth"))
    optimizer_g=torch.optim.Adam(generator.parameters(),lr=0.0002)
    optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=0.0002)
    optimizer_s=torch.optim.Adam(unet.parameters(),lr=0.0002)
    generator.cuda()
    discriminator.cuda()
    unet.cuda()
    EPOCH=200
    num_iter=len(train_loader)
    D_LOSS=[]
    G_LOSS=[]
    S_LOSS=[]
    discriminator.train()
    unet.train()
    LAMBDA_GP = 10
    LAMBDA_CLS = 1
    LAMBDA_REC = 10
    LAMBDA_SEG = 10

    log = pd.DataFrame(index=[], columns=['epoch','lr','d_loss','g_loss','s_loss','d_loss_real','d_loss_fake','d_loss_cls','d_loss_gp','g_loss_fake','g_loss_cls','g_loss_rec','g_loss_seg_'])

    print("launch algo")
    for epoch in range(EPOCH):
        if epoch==25:
            update_lr(optimizer_g,0.0001)
            update_lr(optimizer_d,0.0001)
            update_lr(optimizer_s,0.0001)
            print('*****Change lr to :',optimizer_g.param_groups[0]['lr'])
        elif epoch==50:
            update_lr(optimizer_g,0.00005)
            update_lr(optimizer_d,0.00005)
            update_lr(optimizer_s,0.00005)
            print('*****Change lr to :',optimizer_g.param_groups[0]['lr'])
        elif epoch==75:
            update_lr(optimizer_g,0.00001)
            update_lr(optimizer_d,0.00001)
            update_lr(optimizer_s,0.00001)
            print('***** Change lr to :',optimizer_g.param_groups[0]['lr'])
        elif epoch==100:
            update_lr(optimizer_g,0.00002)
            update_lr(optimizer_d,0.00002)
            update_lr(optimizer_s,0.00002)
            print('***** Change lr to :',optimizer_g.param_groups[0]['lr'])
        elif epoch==125:
            update_lr(optimizer_g,0.00001)
            update_lr(optimizer_d,0.00001)
            update_lr(optimizer_s,0.00001)
            print('***** Change lr to :',optimizer_g.param_groups[0]['lr'])
        elif epoch==150:
            update_lr(optimizer_g,0.00004)
            update_lr(optimizer_d,0.00004)
            update_lr(optimizer_s,0.00004)
            print('***** Change lr to :',optimizer.param_groups[0]['lr'])
        elif epoch==175:
            update_lr(optimizer_g,0.00003)
            update_lr(optimizer_d,0.00003)
            update_lr(optimizer_s,0.00003)
            print('***** Change lr to :',optimizer.param_groups[0]['lr'])
        
        lr_optimizer = optimizer_g.param_groups[0]['lr']
        d_loss_,g_loss_,d_loss_real_,d_loss_cls_,d_loss_fake_,d_loss_gp_,g_loss_fake_,g_loss_cls_,g_loss_rec_,g_loss_seg_,s_loss_=0,0,0,0,0,0,0,0,0,0,0
        
        ##training mode set
        generator.train()
        for i,(t1ce,t18t2f,seg) in enumerate(train_loader):
            #discriminator real
            label_=torch.randint(3,(t1ce.size(0),))
            label = label2onehot(label_,3).cuda()
            real=torch.zeros(t1ce.size(0),t1ce.size(1),t1ce.size(2),t1ce.size(3))
            for i,l in enumerate(label_):
                if l==0:
                    real[i]=t1ce[i]
                else:
                    break
            out_src, out_cls = discriminator(real.float().cuda(), t18t2f.float().cuda())
            d_loss_real = - torch.mean(out_src.sum([1,2,3]))
            d_loss_cls = classification_loss(out_cls, label)

            '''discriminator'''
            fake=generator(t18t2f.float().cuda(),label)
            out_src, out_cls = discriminator(fake.detach(), t18t2f.float().cuda())
            d_loss_fake = torch.mean(out_src.sum([1,2,3]))
            
            # Compute loss for gradient penalty.
            alpha = torch.rand(real.size(0), 1, 1, 1).cuda()
            x_hat = (alpha * real.cuda().data + (1 - alpha) * fake.data).requires_grad_(True)
            out_src, _ = discriminator(x_hat,t18t2f.float().cuda())
            d_loss_gp = gradient_penalty(out_src, x_hat)
            # d_loss_gp.backward(retain_graph=True)
            d_loss=d_loss_real+d_loss_fake+LAMBDA_CLS*d_loss_cls +LAMBDA_GP*d_loss_gp
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()
            
            '''generator '''
            fake = generator(t18t2f.float().cuda(), label)
            out_src,out_cls=discriminator(fake,t18t2f.float().cuda())
            g_loss_fake = -torch.mean(out_src.sum([1,2,3]))
            g_loss_cls = classification_loss(out_cls,label)
            g_loss_rec = torch.mean(torch.abs(real.float().cuda() - fake).sum([1,2,3]))
            pred = unet(fake,label)
            g_loss_seg = dice_loss(pred,seg.float().cuda())
            g_loss = g_loss_fake + LAMBDA_CLS*g_loss_cls +  g_loss_rec*LAMBDA_REC + g_loss_seg*LAMBDA_SEG
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            
            '''segmentor'''
            fake = generator(t18t2f.float().cuda(), label)
            pred = unet(fake.detach(),label)
            pred_2 = unet(real.float().cuda(),label)
            s_loss = dice_loss(pred,seg.float().cuda())
            s_loss_2 = dice_loss(pred_2,seg.float().cuda())
            s_loss = 0.7*s_loss + 0.3 *s_loss_2
            optimizer_s.zero_grad()
            s_loss.backward()
            optimizer_s.step()
            
            d_loss_+=d_loss.data.cpu().numpy()
            g_loss_ += g_loss.data.cpu().numpy()
            d_loss_real_ +=d_loss_real.data.cpu().numpy()
            d_loss_cls_ +=d_loss_cls.data.cpu().numpy()
            d_loss_fake_ +=d_loss_fake.data.cpu().numpy()
            d_loss_gp_ +=d_loss_gp.data.cpu().numpy()
            g_loss_fake_ +=g_loss_fake.data.cpu().numpy()
            g_loss_cls_ +=g_loss_cls.data.cpu().numpy()
            g_loss_rec_+=g_loss_rec.data.cpu().numpy()
            g_loss_seg_+=g_loss_seg.data.cpu().numpy()
            s_loss_ += s_loss.data.cpu().numpy()

        dloss=d_loss_
        gloss=g_loss_
        dlossreal=d_loss_real_
        dlossfake=d_loss_fake_
        dlosscls=d_loss_cls_
        dlossgp=d_loss_gp_
        glossfake=g_loss_fake_
        glosscls=g_loss_cls_
        glossrec=g_loss_rec_
        glossseg=g_loss_seg_
        sloss=s_loss_

        tmp = pd.Series([epoch+1,lr_optimizer,dloss,gloss,sloss,dlossreal,dlossfake,dlosscls,dlossgp,glossfake,glosscls,glossrec,glossseg,],index=['epoch','lr','d_loss','g_loss','s_loss','d_loss_real','d_loss_fake','d_loss_cls','d_loss_gp','g_loss_fake','g_loss_cls','g_loss_rec','g_loss_seg_'])    

        print('EPOCH %d : d_loss = %.4f , g_loss = %.4f , s_loss = %.4f'%(epoch+1,d_loss_/num_iter,g_loss_/num_iter,s_loss_/num_iter))
        print("d_real = %.4f , d_fake = %.4f , d_cls = %.4f , d_gp = %.4f | g_fake = %.4f , g_cls = %.4f , g_rec = %.4f , g_seg = %.4f"%( d_loss_real_/num_iter , d_loss_fake_/num_iter , d_loss_cls_/num_iter , d_loss_gp_/num_iter ,  g_loss_fake_/num_iter , g_loss_cls_/num_iter , g_loss_rec_/num_iter , g_loss_seg_/num_iter) )
        
        log = log.append(tmp, ignore_index=True)
        log.to_csv('E:/Marwan/MISOF/Weights/loggan_miso_f.csv', index=False)

        D_LOSS.append(d_loss_/num_iter)
        G_LOSS.append(g_loss_ / num_iter)
        S_LOSS.append(g_loss_seg_ / num_iter)

        ##test for fixed
        generator.eval()
        c_fix=torch.tensor([[1,0,0]]).float().cuda()
        fix_flair = generator(t18t2f_fix.float().cuda(),c_fix).data.cpu().numpy()
        fix_flair_2 = generator(t18t2f_fix_2.float().cuda(),c_fix).data.cpu().numpy()
        fix_flair_3 = generator(t18t2f_fix_3.float().cuda(),c_fix).data.cpu().numpy()
        
        c_fix=torch.tensor([[0,1,0]]).float().cuda()
        
        c_fix=torch.tensor([[0,0,1]]).float().cuda()
        fix_t1 = generator(t18t2f_fix.float().cuda(),c_fix).data.cpu().numpy()
        fix_t1_2 = generator(t18t2f_fix_2.float().cuda(),c_fix).data.cpu().numpy()
        fix_t1_3 = generator(t18t2f_fix_3.float().cuda(),c_fix).data.cpu().numpy()
        gen_fix = np.hstack((t18t2f_fix[0][0],fix_flair[0][0],fix_t1[0][0],seg_fix[0][0]))
        gen_fix_2 = np.hstack((t18t2f_fix_2[0][0],fix_flair_2[0][0],fix_t1_2[0][0],seg_fix_2[0][0]))
        gen_fix_3 = np.hstack((t18t2f_fix_3[0][0],fix_flair_3[0][0],fix_t1_3[0][0],seg_fix_3[0][0]))

    model_save_g='E:/Marwan/MISOF/Weights/miso_f_generator_t2_tumor_bw.pth'
    torch.save(generator.state_dict(), model_save_g)
    model_save_s='E:/Marwan/MISOF/Weights/miso_f_segmentor_t2_tumor_bw.pth'
    torch.save(unet.state_dict(), model_save_s)
    model_save_d='E:/Marwan/MISOF/Weights/miso_f_discriminator_t2_bw.pth'
    torch.save(discriminator.state_dict(), model_save_d)


if __name__ == '__main__':
    main()