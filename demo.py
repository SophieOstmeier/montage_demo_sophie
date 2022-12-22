import os
import cv2
from regutils import simpleelastix_utils as sutl
from dicomutils import imageutils
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
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from regutils.simpleelastix_utils import reorder_yxz


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

def img_rescale_0_1(matin_img, range= None):
    matin = sitk.GetArrayFromImage(matin_img)
    # mat_xyz = reorder_yxz(matin).copy()
    if range:
        mat_xyz = imageutils.imrescale(matin, range, [0, 1])
    else:
        mat_xyz = imageutils.imrescale(matin, [], [0, 1])

    return mat_xyz

def montage_loop(path_GT_1,path_GT_2,path_GT_3,path_NCCT,path_softpred,path_montages_3, path_df):
    # dataframe
    df = pandas.read_excel(path_df, index_col="('reference',)", sheet_name='all', engine='openpyxl')
    df2 = df.sort_index(axis=0)
    dice = df2["('1', 'Dice')"].tolist()

    volume = df2["('1', 'Volume Reference')"].tolist()
    # df_id = df["('reference',)"].tolist()


    # for glob.glob lists
    GT_1 = natsorted(glob.glob(str(path_GT_1 + '/*.nii.gz')))
    GT_2 = natsorted(glob.glob(str(path_GT_2 + '/*.nii.gz')))
    GT_3 = natsorted(glob.glob(str(path_GT_3 + '/*.nii.gz')))
    pred_ = natsorted(glob.glob(str(path_pred + '/*.nii.gz')))
    NCCT_ = natsorted(glob.glob(str(path_NCCT + '/*.nii.gz')))
    sofpred_ = natsorted(glob.glob(str(path_softpred + '/*.npz')))
    # softpred_id = [encoded_id_NCCT = i.split('/')[-1] for i in sofpred_]
    # encoded_num_NCCT = re.findall('([0-9]+)', encoded_id_NCCT)[0]
    assert len(GT_1)==len(GT_2)==len(GT_3)==len(NCCT_)==len(sofpred_)==len(dice)==len(volume), 'different list lengths' #==len(df_id)

    for i1,i2,i3,gt,sp,p,di,vol in zip(GT_1,GT_2,GT_3,NCCT_,sofpred_, pred_, dice, volume): #df_id
        encoded_id = gt.split('/')[-1]
        encoded_num = re.findall('([0-9]+)', encoded_id)[0]

        # encoded_id_df = df_id_i.split('/')[-1]
        # encoded_num_df = re.findall('([0-9]+)', encoded_id_df)[0]
        # assert encoded_num == encoded_num_df, 'different cases in file and df'
        GT_1 = sitk.ReadImage(os.path.join(path_GT_1, i1))
        GT_2 = sitk.ReadImage(os.path.join(path_GT_2, i2))
        GT_3 = sitk.ReadImage(os.path.join(path_GT_3, i3))
        NCCT = sitk.ReadImage(os.path.join(path_NCCT, gt))
        pred = sitk.ReadImage(os.path.join(path_pred, p))
        softpred = np.load(os.path.join(path_softpred, sp))['softmax']# Why ['softmax']?

        text = 'dice' + str(round(di, 2)) + '_volume' + str(round(vol, 2)) # + '_volume' + str(
            # round(vol, 0)) + 'ml'

    # GT = sitk.ReadImage("NCCT_001_gt.nii.gz")
    # NCCT = sitk.ReadImage("NCCT_001_0000_NCCT.nii.gz")
    # softpred = np.load("NCCT_001_softmax.npz")['softmax']

        #FOCUS ON LABEL CLASS
        p1 = softpred[1]

        #CREATE RGB MAGES - MONTAGE STYLE
        #if 1:
            #ncct_rgb = sutl.sitk2montage(NCCT, str(encoded_num + '_NCCT.png'),range=[0,60])
            #sutl.sitk2montage(NCCT,str(encoded_num + '_GT.png'),range=[0,60],maskovl=GT)
            #sutl.sitk2montage(NCCT,str(encoded_num + '_pred.png'),range=[0,60],maskovl=pred,mask_mix=[255,255,0])
            #blended_rgb = softpred_to_rgb(p1,ncct_rgb)
            #imageio.imwrite(str(encoded_num + '_softmax.png'),(blended_rgb*255).astype(np.uint8))

        # metrics for image

        #NOW LETS MAKE THE MORE SOFISTICATED ROW DEPICTION
        #lets just show slices with either pred:20% or a GT lesion
        sum_array = sitk.GetArrayFromImage(GT_1) + sitk.GetArrayFromImage(GT_2) + sitk.GetArrayFromImage(GT_3)
        pred_of_interest = np.logical_and(p1 > 0.2, sum_array)
        slices_no_interest = np.argwhere(np.sum(pred_of_interest,axis=(1,2))==0)
        # added to original for valid_slices_count = 0 change np.logical_and( to np.logical_or(
        if pred_of_interest.shape[0] == slices_no_interest.shape[0]:
            pred_of_interest = np.logical_or(p1 > 0.2, sitk.GetArrayFromImage(pred))
            slices_no_interest = np.argwhere(np.sum(pred_of_interest, axis=(1, 2)) == 0)
        NCCTmask = sitk.GetArrayFromImage(NCCT) > 100
        if slices_no_interest.shape[0] == p1.shape[0]:
            continue
            # middle = p1.shape[0]//2
            # min = middle + middle/2
            # max = middle + middle/2
        for k in slices_no_interest:
            NCCTmask[k[0]] = 0

        valid_slices_count=(np.sum(NCCTmask,axis=(1,2))>0).sum()

        if valid_slices_count == 0:
            pred_of_interest.shape[0]

        # img and masks for multilabel
        # sum = []
        # for i in zip(list(GT_1,GT_2,GT_3)):
        #     array = sitk.GetArrayFromImage(i)
        #     sum.append(array)
        GT_1_mask = sitk.GetArrayFromImage(GT_1)
        GT_2_mask = sitk.GetArrayFromImage(GT_2)
        GT_3_mask = sitk.GetArrayFromImage(GT_3)
        assert GT_1_mask.shape == GT_2_mask.shape == GT_3_mask.shape, 'different dimensions'
        sum_seg = (GT_1_mask + GT_2_mask + GT_3_mask)/3
        NCCT_rescale = img_rescale_0_1(NCCT, range=[0,60])
        CT_rgb = imageutils.stack2rgbmont(NCCT_rescale, [1,1,1])
        overlay = imageutils.stack2rgbmont(sum_seg , [1, 0, 0])

        A = sutl.sitk2montage(NCCT, str(encoded_num +'_ROW_NCCT.png'),range=[0,60],cropmask=NCCTmask,rc=[1,valid_slices_count])
        B = sutl.sitk2montage(NCCT,str(encoded_num +'_ROW_GT_1.png'),range=[0,60],maskovl=GT_1,mask_mix=[255,0,0],cropmask=NCCTmask,rc=[1,valid_slices_count])
        C = sutl.sitk2montage(NCCT, str(encoded_num + '_ROW_GT_2.png'), range=[0, 60], maskovl=GT_2,mask_mix=[0,255,0], cropmask=NCCTmask,
                              rc=[1, valid_slices_count])
        D = sutl.sitk2montage(NCCT, str(encoded_num + '_ROW_GT_3.png'), range=[0, 60], maskovl=GT_3,mask_mix=[0,0,255], cropmask=NCCTmask,
                              rc=[1, valid_slices_count])
        # E = imageutils.rgbmaskonrgb(CT_rgb, overlay)
        F = sutl.sitk2montage(NCCT,str(encoded_num +'_ROW_pred.png'),range=[0,60],maskovl=pred,mask_mix=[255,255,0],cropmask=NCCTmask,rc=[1,valid_slices_count])
        G = (255*softpred_to_rgb(p1,A,cropmask=NCCTmask,rc=[1,valid_slices_count])).astype(np.uint8)
        imageio.imwrite(str(encoded_num + '_ROW_softmax.png'),D)

        vcat = np.concatenate((A,B,C,D,F,G),axis=0)
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
    # READ IMAGES # path to directories
    path_NCCT = '/Volumes/T7/NCCT_project_ncctROI/3_Round_Ben/interrater_analysis_1_2/fold_0_validation_images'
    path_GT_1 = '/Volumes/T7/NCCT_project_ncctROI/3_Round_Ben/interrater_analysis_1_2/fold_0_validation_Ben'
    path_GT_2 = '/Volumes/T7/NCCT_project_ncctROI/3_Round_Ben/interrater_analysis_1_2/fold_0_validation_Abdel'
    path_GT_3 = '/Volumes/T7/NCCT_project_ncctROI/3_Round_Ben/interrater_analysis_1_2/fold_0_validation_Jeremy'
    path_pred = '/Users/sophieostmeier/Desktop/DLmachine/NCCTfolders/nnUNet_trained_models/nnUNet/3d_fullres/Task001_NCCT/nnUNetTrainerV2_random_data_loader__nnUNetPlansv2.1/fold_0/validation_raw_postprocessed'
    path_softpred = '/Users/sophieostmeier/Desktop/DLmachine/NCCTfolders/nnUNet_trained_models/nnUNet/3d_fullres/Task001_NCCT/nnUNetTrainerV2_random_data_loader__nnUNetPlansv2.1/fold_0/validation_raw'
    path_montages_0 = '/Volumes/T7/NCCT_project_ncctROI/3D_JSONs/montages/dice0'
    path_montages_7 = '/Volumes/T7/NCCT_project_ncctROI/3D_JSONs/montages/dice7'
    path_montages_3 = '/Users/sophieostmeier/Desktop/random_rater'
    pred = 0.5
    path_df = '/Users/sophieostmeier/Desktop/DLmachine/NCCTfolders/nnUNet_trained_models/nnUNet/3d_fullres/Task001_NCCT/nnUNetTrainerV2_random_data_loader__nnUNetPlansv2.1/fold_0/validation_raw_postprocessed/summary.xlsx'
    maybe_mkdir_p(path_montages_3)

    montage_loop(path_GT_1,path_GT_2,path_GT_3,path_NCCT,path_softpred,path_montages_3, path_df)