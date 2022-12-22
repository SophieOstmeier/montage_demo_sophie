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
import PIL
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

def montage_loop(path_B,path_C,path_D, path_E,path_NCCT,path_montages, path_df=None):

    if not os.path.exists(path_montages):
        os.mkdir(path_montages)

    # dataframe
    if not path_df is None:
        df = pandas.read_excel(path_df, sheet_name='all', index_col="('reference',)", engine='openpyxl')
        vol_reference = df["('1', 'Volume Test')"].tolist()


    # for glob.glob lists
    D_ = natsorted(glob.glob(str(path_D + '/*.npz')))
    id = []
    for i in D_:
        id_num = i.split('/')[-1].split('.')[0] + '.nii.gz'
        id.append(id_num)
    B_ = []
    for i in natsorted(glob.glob(str(path_B+ '/*.nii.gz'))):
        id_num_b = i.split('/')[-1].split('.')[0] + '.nii.gz'
        if id_num_b in id:
            B_.append(i)
    C_ = []
    for i in natsorted(glob.glob(str(path_C+ '/*.nii.gz'))):
        id_num_b = i.split('/')[-1].split('.')[0] + '.nii.gz'
        if id_num_b in id:
            C_.append(i)

    E_ = []
    for i in natsorted(glob.glob(str(path_E+ '/*.nii.gz'))):
        id_num_b = i.split('/')[-1].split('.')[0] + '.nii.gz'
        if id_num_b in id:
            E_.append(i)

    NCCT_ = []
    for i in natsorted(glob.glob(str(path_NCCT+ '/*.nii.gz'))):
        id_num_n = i.split('/')[-1].rsplit('_',1)[0].split('.')[0] + '.nii.gz'
        if id_num_n in id:
            NCCT_.append(i)
    count = 0


    #assert len(B_) == len(C_) == len(D_)== len(vol_reference), 'different list lengths'
    for b, c, d, e, a, vol in zip(B_, C_, D_,E_,NCCT_,vol_reference):
        # if di < 1.0:
        count += 1
        encoded_id = b.split('/')[-1]
        encoded_num = re.findall('([0-9]+)', encoded_id)[0]+'_'+ str(round(vol,2)) +'ml'
        B = sitk.ReadImage(b)
        C = sitk.ReadImage(c)
        D = np.load(os.path.join(path_D, d))['softmax'] # Why ['softmax']?
        E = sitk.ReadImage(e)
        NCCT = sitk.ReadImage(a)

        p1 = D[1]

        # CREATE RGB MAGES - MONTAGE STYLE
        # if 1:
        #     ncct_rgb = sutl.sitk2montage(NCCT, str(encoded_num + '_NCCT.png'), range=[0, 60])
        #     B_rgb = sutl.sitk2montage(NCCT, str(encoded_num + '_B.png'), range=[0, 60], maskovl=B)
        #     blended_rgb = softpred_to_rgb(p1, ncct_rgb)
        #     imageio.imwrite(str(encoded_num + '_softmax.png'),(blended_rgb*255).astype(np.uint8))

        # metrics for image

        # NOW LETS MAKE THE MORE SOFISTICATED ROW DEPICTION
        # lets just show slices with either pred:20% or a B lesion
        pred_of_interest = sitk.GetArrayFromImage(B)
        slices_no_interest = np.argwhere(np.sum(pred_of_interest, axis=(1, 2)) == 0)
        NCCTmask = sitk.GetArrayFromImage(NCCT) > 100
        for k in slices_no_interest:
            NCCTmask[k[0]] = 0

        valid_slices_count = (np.sum(NCCTmask, axis=(1, 2)) > 0).sum()

        if valid_slices_count > 0:
            A_ = sutl.sitk2montage(NCCT, str(encoded_num + '_ROW_NCCT.png'), range=[0, 60], cropmask=NCCTmask,
                                  rc=[1, valid_slices_count])
            E_ = sutl.sitk2montage(NCCT, str(encoded_num + '_ROW_E.png'), range=[0, 60], maskovl=E,
                                  cropmask=NCCTmask, rc=[1, valid_slices_count])
            B_ = sutl.sitk2montage(NCCT, str(encoded_num + '_ROW_B.png'), range=[0, 60], maskovl=B,
                                  cropmask=NCCTmask, rc=[1, valid_slices_count], mask_mix=[0,255,255])
            C_ = sutl.sitk2montage(NCCT, str(encoded_num + '_ROW_C.png'), range=[0, 60], maskovl=C,
                                  cropmask=NCCTmask, rc=[1, valid_slices_count], mask_mix=[255,255,0])
            D_ = (255 * softpred_to_rgb(p1, A_, cropmask=NCCTmask, rc=[1, valid_slices_count])).astype(np.uint8)
            imageio.imwrite(str(encoded_num + '_ROW_softmax.png'), D_)

            # dice_score = dice(test=None, reference=B, confusion_matrix=None, nan_for_nonexisting=True)
            # sensitivity_ = sensitivity(test=None, reference=B, confusion_matrix=None, nan_for_nonexisting=True)
            # TP_ = TP(test=None, reference=B, confusion_matrix=None, nan_for_nonexisting=True)

            vcat = np.concatenate((A_,E_, B_, C_,D_), axis=0)
            # original: vcat = np.concatenate( (A,B,C,(blended_rgb*255).astype(np.uint8)),axis=0)
            # path_montage_3 = path_montages_3 + '/' + encoded_num + '_montage_' + text + '.png'
            # imageio.imwrite(path_montage_3,vcat)
            path_montage = path_montages + '/' + encoded_num + '.png'
            imageio.imwrite(path_montage, vcat)

if __name__=="__main__":
    # READ IMAGES # path to directories
    path_B = '/Volumes/T7/Publications/NCCT_paper_DL/Data/labelsTr_Jeremy_227'
    path_C = '/Volumes/T7/Publications/NCCT_paper_DL/Data/labelsTr_Abdel_227'
    path_D = '/Users/sophieostmeier/Desktop/todayyy/NCCTfolders/nnUNet_trained_models/nnUNet/3d_fullres/Task075_NCCT_227_mirror_Ben/nnUNetTrainerV2_mirror_227__nnUNetPlansv2.1/all_npz'
    path_E = '/Volumes/T7/Publications/NCCT_paper_DL/Data/labelsTr_Ben_227'
    path_NCCT = '/Users/sophieostmeier/Desktop/todayyy/NCCTfolders/nnUNet_raw/nnUNet_raw_data/Task005_NCCT_no_mirror_227_Ben/imagesTr'
    path_montages = '/Volumes/T7/Publications/NCCT_paper_DL/montages_3rater'

    path_df = '/Volumes/T7/Publications/NCCT_paper_DL/configurations/summary_mirror_227.xlsx'

    montage_loop(path_B, path_C,path_D,path_E, path_NCCT, path_montages,path_df)