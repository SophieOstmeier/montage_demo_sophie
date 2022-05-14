import os
import cv2
from regutils import simpleelastix_utils as sutl
import SimpleITK as sitk
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from dicomutils.imageutils import montage, get_mask_bounds_zyx
import imageio
from natsort import natsorted
import glob
import re
import pandas
import xlrd


def softpred_to_rgb(imgin,ncct_rgb,cropmask=None,rc=None):
    # MAKE SOFT PRED INTO RGB COLOR MAP WITH JET SCALE
    if cropmask is not None:
        rowmin, rowmax, colmin, colmax, slicemin, slicemax = get_mask_bounds_zyx(cropmask)
        imgin = imgin[slicemin:slicemax + 1, colmin:colmax + 1, rowmin:rowmax + 1]
        rc = [rc[0],int(imgin.shape[0]/rc[0]) ]

    p1_mont = montage(sutl.reorder_yxz(imgin),rc=rc)
    cmap = plt.get_cmap('jet')
    p1_mont_rgb = cmap(p1_mont)[:, :, 0:3]
    # mask out less than 0.05
    p1_mont_rgb[p1_mont < 0.05] = 0

    # MERGE COLORMAP WITH BG TRANSPARENCY
    alpha = np.expand_dims((p1_mont > 0.05).astype(float) * 0.8, axis=2)
    blended_rgb = (ncct_rgb.astype(float) / 255.0 * (1 - alpha) + p1_mont_rgb * alpha)
    return blended_rgb

def make_colorbar():
    height = 1200
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

    pixelWH = fig.canvas.get_width_height()
    plt.savefig("colmap.png",dpi = 100*(height/pixelWH[1]),facecolor=fig.get_facecolor(),transparent=False)
    plt.close()
    # colbar = imageio.imread("colmap.png",)
    # colbar_pad = np.zeros( (*imgin.shape[0:2],4),dtype=np.uint8 )
    # colbar_pad[:,-colbar.shape[1]:,:] = colbar
    # colbar_pad = colbar_pad.astype(float)/255
    # alpha = np.expand_dims(colbar_pad[:,:,3],axis=2)
    # return imgin * (1-alpha) + colbar_pad[:,:,0:3] *alpha

#CREATE SEPERATE COLORBAR IMG FOR REFERENCE
blended_rgb = make_colorbar()

#READ IMAGES # path to directories
path_GT = '/Volumes/T7/NCCT_project_ncctROI/1_Round_Ben/NCCTfolders/nnUNet_raw/nnUNet_raw_data/Task001_NCCT/labelsTr'
path_NCCT = '/Volumes/T7/NCCT_project_ncctROI/1_Round_Ben/NCCTfolders/nnUNet_raw/nnUNet_raw_data/Task001_NCCT/imagesTr'
path_softpred = '/Volumes/T7/NCCT_project_ncctROI/1_Round_Ben/NCCTfolders/nnUNet_trained_models/nnUNet/3d_fullres/Task001_NCCT/nnUNetTrainerV2__nnUNetPlansv2.1/validation_raw_all'
path_montages_0 = '/Volumes/T7/NCCT_project_ncctROI/3D_JSONs/montages/dice0'
path_montages_7 = '/Volumes/T7/NCCT_project_ncctROI/3D_JSONs/montages/dice7'
path_montages_3 = '/Volumes/T7/NCCT_project_ncctROI/1_Round_Ben/3D_JSONs/montages/dice3'
pred = 0.5
path_df = '/Volumes/T7/NCCT_project_ncctROI/1_Round_Ben/3D_JSONs/3D_summary_combined.xlsx'


def montage_loop(path_GT,path_NCCT,path_softpred,path_montages_3, path_df):
    # dataframe
    df = pandas.read_excel(path_df, sheet_name='summary_0-4', index_col='reference', engine='openpyxl')
    dice = df['Dice'].tolist()
    sensitivity = df['Recall'].tolist()
    volume = df['volume reference'].tolist()


    # for glob.glob lists
    GT_ = natsorted(glob.glob(str(path_GT+ '/*.nii.gz')))
    NCCT_ = natsorted(glob.glob(str(path_NCCT+ '/*.nii.gz')))
    sofpred_ = natsorted(glob.glob(str(path_softpred+ '/*.npz')))
    assert len(GT_)==len(NCCT_)==len(sofpred_)==len(dice)==len(sensitivity), 'different list lengths'

    for i,f,g,di,se,vol in zip(GT_,NCCT_,sofpred_, dice, sensitivity, volume):
        encoded_id = i.split('/')[-1]
        encoded_num = re.findall('([0-9]+)', encoded_id)[0]
        if not encoded_num == '210':
            continue
        GT = sitk.ReadImage(os.path.join(path_GT, i))
        NCCT = sitk.ReadImage(os.path.join(path_NCCT, f))
        softpred = np.load(os.path.join(path_softpred, g))['softmax']# Why ['softmax']?

        text = 'dice' + str(round(di, 2)) + '_sensitivity' + str(round(se, 2)) + '_volume' + str(
            round(vol, 0)) + 'ml'

    # GT = sitk.ReadImage("NCCT_001_gt.nii.gz")
    # NCCT = sitk.ReadImage("NCCT_001_0000_NCCT.nii.gz")
    # softpred = np.load("NCCT_001_softmax.npz")['softmax']

        #FOCUS ON LABEL CLASS
        p1 = softpred[1]

        #CREATE RGB MAGES - MONTAGE STYLE
        if 1:
            ncct_rgb = sutl.sitk2montage(NCCT, str(encoded_num + '_NCCT.png'),range=[0,60])
            sutl.sitk2montage(NCCT,str(encoded_num + '_GT.png'),range=[0,60],maskovl=GT)
            sutl.sitk2montage(NCCT,str(encoded_num + '_pred.png'),range=[0,60],maskovl=(p1>0.5).astype(np.uint8),mask_mix=[255,255,0])
            blended_rgb = softpred_to_rgb(p1,ncct_rgb)
            imageio.imwrite(str(encoded_num + '_softmax.png'),(blended_rgb*255).astype(np.uint8))

        # metrics for image

        #NOW LETS MAKE THE MORE SOFISTICATED ROW DEPICTION
        #lets just show slices with either pred:20% or a GT lesion
        pred_of_interest = np.logical_and(p1>0.2,sitk.GetArrayFromImage(GT))
        slices_no_interest = np.argwhere(np.sum(pred_of_interest,axis=(1,2))==0)
        # added to original for valid_slices_count = 0 change np.logical_and( to np.logical_or(
        if pred_of_interest.shape[0] == slices_no_interest.shape[0]:
            pred_of_interest = np.logical_or(p1 > 0.2, sitk.GetArrayFromImage(GT))
            slices_no_interest = np.argwhere(np.sum(pred_of_interest, axis=(1, 2)) == 0)
        NCCTmask = sitk.GetArrayFromImage(NCCT)>100
        for k in slices_no_interest:
            NCCTmask[k[0]] = 0

        valid_slices_count=(np.sum(NCCTmask,axis=(1,2))>0).sum()

        A = sutl.sitk2montage(NCCT, str(encoded_num +'_ROW_NCCT.png'),range=[0,60],cropmask=NCCTmask,rc=[1,valid_slices_count])
        B = sutl.sitk2montage(NCCT,str(encoded_num +'_ROW_GT.png'),range=[0,60],maskovl=GT,cropmask=NCCTmask,rc=[1,valid_slices_count])
        C = sutl.sitk2montage(NCCT,str(encoded_num +'_ROW_pred.png'),range=[0,60],maskovl=(p1>0.5).astype(np.uint8),mask_mix=[255,255,0],cropmask=NCCTmask,rc=[1,valid_slices_count])
        D = (255*softpred_to_rgb(p1,A,cropmask=NCCTmask,rc=[1,valid_slices_count])).astype(np.uint8)
        imageio.imwrite(str(encoded_num + '_ROW_softmax.png'),D)

        vcat = np.concatenate( (A,B,C,D),axis=0)
        # did not workk vcat = np.concatenate( (A,B,C,(blended_rgb*255).astype(np.uint8)),axis=0)
        path_montage_3 = path_montages_3 + '/' + encoded_num + '_montage_' + text + '.png'
        imageio.imwrite(path_montage_3,vcat)
        # if di == 0:
        #     path_montage_0 = path_montages_0 + '/' + encoded_num + '_montage_' + text + '.png'
        #     imageio.imwrite(path_montage_0,vcat)
        # else:
        #     path_montage_7 = path_montages_7 + '/' + encoded_num + '_montage_' + text + '.png'
        #     imageio.imwrite(path_montage_7, vcat)
        # add metrics to image
        # fontScale = 1.5
        # thickness = 4
        # image = cv2.imread(path_montage)
        # text = 'Dice: ' + str(round(di, 2)) + ', Sensitivity: ' + str(round(se, 2)) + ' GT volume: ' + str(round(vol, 2)) + 'ml'
        # org = (50 , round((image.shape[0] * 0.25)))
        # cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 64, 234), thickness)
        # cv2.imwrite(path_montage, image)

if __name__=="__main__":
    montage_loop(path_GT,path_NCCT,path_softpred,path_montages_3, path_df)