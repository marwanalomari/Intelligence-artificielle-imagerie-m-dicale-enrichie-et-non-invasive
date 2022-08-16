# Marwan AL OMARI, Master 2 Engineering of connected objects
# University of Poitiers
# marwanalomari@yahoo.com
# Internship subject: Artificial intelligence for enhanced non-invasive medical imaging
# Internship location: University of Poitiers, XLIM (CNRS 7252) and I3M (CHU Poitiers) laboratories 

'''Please cite the work code in case of use, thanks!'''

# Import libraries
import numpy as np
import torch

def dice_loss(m1, m2):
    num = m1.size(0)
    m1  = m1.view(num,-1)
    m2  = m2.view(num,-1)
    intersection = (m1 * m2)
    scores = (2. * intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    score = scores.sum()/num
    return 1-score


def dice_score(m1, m2):
    num = m1.size(0)
    m1  = m1.view(num,-1)
    m2  = m2.view(num,-1)
    intersection = (m1 * m2)
    scores = (2. * intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    return scores.sum()/num


if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.rand((4, 1, 128, 128))
    y = torch.rand((4, 1, 128, 128))
    print(dice_score(x,y))