from regutils import simpleelastix_utils as sutl
import SimpleITK as sitk
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from dicomutils.imageutils import montage
import imageio
import io

def make_colorbar(height):
    plt.rcParams['toolbar'] = 'None'
    fig, ax = plt.subplots(figsize=(2, 12))
    fig.set_dpi(100)
    dpi = fig.dpi
    fig.subplots_adjust(right=0.5)
    fig.set_facecolor([0, 0, 0])
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='vertical')
    plt.setp(plt.getp(ax, 'yticklabels'), color=[1,1,1,1])

    #pixelWH = fig.canvas.get_width_height()
    plt.savefig("colmap.png",backend="AGG",)
    plt.close()
    return imageio.imread("colmap.png")

make_colorbar(3000)
#READ IMAGES
GT = sitk.ReadImage("NCCT_001_gt.nii.gz")
NCCT = sitk.ReadImage("NCCT_001_0000_NCCT.nii.gz")
softpred = np.load("NCCT_001_softmax.npz")['softmax']

#FOCUS ON LABEL CLASS
p1 = softpred[1]

#CREATE RGB
ncct_rgb = sutl.sitk2montage(NCCT,'001_A.png',range=[0,60])
sutl.sitk2montage(NCCT,'001_B.png',range=[0,60],maskovl=GT)

#MAKE SOFT PRED INTO RGB COLOR MAP WITH JET SCALE
p1_mont = montage(p1,zyx=True)
cmap = plt.get_cmap('jet')
p1_mont_rgb = cmap(p1_mont)[:,:,0:3]
#mask out less than 0.05
p1_mont_rgb[p1_mont<0.05] = 0

#MERGE COLORMAP WITH BG TRANSPARENCY
alpha = np.expand_dims((p1_mont>0.05).astype(float)*0.8,axis=2)
blended_rgb = (ncct_rgb.astype(float)/255.0*(1-alpha)+p1_mont_rgb*alpha)
#CREATE COLORBAR
cbarimg = make_colorbar(blended_rgb.shape[0])

imageio.imwrite('001_C.png',(blended_rgb*255).astype(np.uint8))

