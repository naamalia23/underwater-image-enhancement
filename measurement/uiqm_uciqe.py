'''
Metrics for unferwater image quality evaluation.

Author: Xuelei Chen 
Email: chenxuelei@hotmail.com

Usage:
python evaluate.py RESULT_PATH

This version we have changed the library skimage to cv2 for image processing
because we can only use opencv library in our project
AND depreceated method like np.int, np.float to its self int,float
'''
import cv2
import numpy as np
import math
# from skimage.measure import compare_psnr, compare_ssim
# import sys
# from skimage import io, color, filters
# import os

def nmetrics(image):
    rgb = image
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    l = lab[:,:,0]

    #1st term
    chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc)**2))**0.5

    #2nd term
    top = int(np.round(0.01*l.shape[0]*l.shape[1]))
    sl = np.sort(l,axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[:top])-np.mean(sl[:top])

    #3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0: satur.append(0)
        elif l1[i] == 0: satur.append(0)
        else: satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    #1st term UICM
    rg = rgb[:,:,0] - rgb[:,:,1]
    yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
    rgl = np.sort(rg,axis=None)
    ybl = np.sort(yb,axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = int(al1 * len(rgl))
    T2 = int(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr- uyb) ** 2)

    uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    #2nd term UISM (k1k2=8x8)
    Rsobelx = rgb[:,:,0] * cv2.Sobel(rgb[:,:,0], ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    Rsobely = rgb[:,:,0] * cv2.Sobel(rgb[:,:,0], ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    Rsobelx = np.uint8(np.absolute(Rsobelx))
    Rsobely = np.uint8(np.absolute(Rsobely))

    Gsobelx = rgb[:,:,1] * cv2.Sobel(rgb[:,:,1], ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    Gsobely = rgb[:,:,1] * cv2.Sobel(rgb[:,:,1], ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    Gsobelx = np.uint8(np.absolute(Gsobelx))
    Gsobely = np.uint8(np.absolute(Gsobely))

    Bsobelx = rgb[:,:,2] * cv2.Sobel(rgb[:,:,2], ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    Bsobely = rgb[:,:,2] * cv2.Sobel(rgb[:,:,2], ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    Bsobelx = np.uint8(np.absolute(Bsobelx))
    Bsobely = np.uint8(np.absolute(Bsobely))
    
    Rsobel = cv2.bitwise_or(Rsobelx, Rsobely)
    Gsobel = cv2.bitwise_or(Gsobelx, Gsobely)
    Bsobel = cv2.bitwise_or(Bsobelx, Bsobely)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    #3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    return uiqm,uciqe

def eme(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]

            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0: blockmin+=1
            if blockmax == 0: blockmax+=1
            eme += w * math.log(blockmax / blockmin)
    return eme

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]
            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            top = plipsub(blockmax,blockmin)
            bottom = plipsum(blockmax,blockmin)

            m = top/bottom
            if m ==0.:
                s+=0
            else:
                s += (m) * np.log(m)

    return plipmult(w,s)

# def main():
#     result_path = sys.argv[1]

#     result_dirs = os.listdir(result_path)

#     sumuiqm, sumuciqe = 0.,0.

#     N=0
#     for imgdir in result_dirs:
#         if '.png' in imgdir:
#             #corrected image
#             corrected = io.imread(os.path.join(result_path,imgdir))

#             uiqm,uciqe = nmetrics(corrected)

#             sumuiqm += uiqm
#             sumuciqe += uciqe
#             N +=1

#             with open(os.path.join(result_path,'metrics.txt'), 'a') as f:
#                 f.write('{}: uiqm={} uciqe={}\n'.format(imgdir,uiqm,uciqe))

#     muiqm = sumuiqm/N
#     muciqe = sumuciqe/N

#     with open(os.path.join(result_path,'metrics.txt'), 'a') as f:
#         f.write('Average: uiqm={} uciqe={}\n'.format(muiqm, muciqe))

# if __name__ == '__main__':
#     main()