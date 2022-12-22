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

def montage_loop(path_list ,path_NCCT, path_montages_0, path_df=None):

    # for glob.glob lists
    path_1 = natsorted(glob.glob(str(path_list[0]+ '/*.nii.gz')))
    path_2 = natsorted(glob.glob(str(path_list[1] + '/*.nii.gz')))
    path_3 = natsorted(glob.glob(str(path_list[2] + '/*.nii.gz')))
    NCCT_ = natsorted(glob.glob(str(path_NCCT + '/*.nii.gz')))
    count = 0

    # dataframe
    text = None
    if not path_df is None:
        df = pandas.read_excel(path_df, sheet_name='summary_0-4', index_col='reference', engine='openpyxl')
        vol_path_1 = df['volume reference'].tolist()
        vol_path_2 = df['volume test rater 2'].tolist()
        vol_path_3 = df['volume test rater 3'].tolist()
        text = 'B_' + str(round(vol_path_1, 2)) + 'A_' + str(round(vol_path_2, 2)) + 'J_' + str(round(vol_path_3, 2))
        # assert len(NCCT_) == len(Ben_) == len(Abdel_) == len(Jeremy_) == len(vol_Ben) == len(vol_Abdel) == len(vol_Jeremy), 'different list lengths'
    else:
        assert len(path_1) == len(path_2) == len(path_3) == len(NCCT_), 'different list lengths'

    for n, one, two, three in zip(NCCT_, path_1, path_2, path_3):
        count += 1
        encoded_id = n.split('/')[-1]
        encoded_num = re.findall('([0-9]+)', encoded_id)[0]
        NCCT = sitk.ReadImage(n)
        expert_1 = sitk.ReadImage(one)
        expert_2 = sitk.ReadImage(two)
        expert_3 = sitk.ReadImage(three)
        # sitk.WriteImage(Heatmap,'/Volumes/T7/NCCT_project_ncctROI/0_interrater_analysis/montages/heatmap.nii')

        # CREATE RGB MAGES - MONTAGE STYLE
        if 1:
            ncct_rgb = sutl.sitk2montage(NCCT, str(encoded_num + '_NCCT.png'), range=[0, 60])
            expert_1_rgb = sutl.sitk2montage(NCCT, str(encoded_num + '_expert_1.png'), range=[0, 60], maskovl=[expert_1])
            expert_2_rgb = sutl.sitk2montage(NCCT, str(encoded_num + '_expert_2.png'), range=[0, 60], maskovl=[expert_2], mask_mix=[255,255,0])
            expert_3_rgb = sutl.sitk2montage(NCCT, str(encoded_num + '_expert_3.png'), range=[0, 60], maskovl=[expert_3], mask_mix=[255,0,0])


        # NOW LETS MAKE THE MORE SOFISTICATED ROW DEPICTION
        # lets just show slices with either pred:20% or a GT lesion
        slices_no_interest = np.argwhere(np.sum(sum_array, axis=(1, 2)) == 0)
        NCCTmask = sitk.GetArrayFromImage(NCCT) > 100
        for k in slices_no_interest:
            NCCTmask[k[0]] = 0

        valid_slices_count = (np.sum(NCCTmask, axis=(1, 2)) > 0).sum()

        if valid_slices_count > 0:
            A = sutl.sitk2montage(NCCT, str(encoded_num + '_ROW_NCCT.png'), range=[0, 60], cropmask=NCCTmask,
                                  rc=[1, valid_slices_count])
            B = sutl.sitk2montage(NCCT, str(encoded_num + '_ROW_expert_1.png'), range=[0, 60], maskovl=expert_1, mask_mix=[0, 218, 0],
                                  cropmask=NCCTmask, rc=[1, valid_slices_count])
            C = sutl.sitk2montage(NCCT, str(encoded_num + '_ROW_expert_2.png'), range=[0, 60], maskovl=expert_2, mask_mix=[255, 105, 224],
                                  cropmask=NCCTmask, rc=[1, valid_slices_count])
            D = sutl.sitk2montage(NCCT, str(encoded_num + '_ROW_expert_3.png'), range=[0, 60], maskovl=expert_3, mask_mix=[255,100,0],
                                  cropmask=NCCTmask, rc=[1, valid_slices_count])


            vcat = np.concatenate((A, B, C, D), axis=0)
            # original: vcat = np.concatenate( (A,B,C,(blended_rgb*255).astype(np.uint8)),axis=0)
            # path_montage_3 = path_montages_3 + '/' + encoded_num + '_montage_' + text + '.png'
            # imageio.imwrite(path_montage_3,vcat)
            if text is not None:
                path_montage_0 = path_montages_0 + '/' + encoded_num + '_montage_' + text + '.png'
            else:
                path_montage_0 = path_montages_0 + '/' + encoded_num + '_montage.png'
            imageio.imwrite(path_montage_0, vcat)


if __name__=="__main__":
    # READ IMAGES # path to directories
    path_Ben = '/Users/sophieostmeier/Desktop/Publications/NCCT_paper_DL/Data/labelsTr_Ben_200'
    path_Jeremy = '/Users/sophieostmeier/Desktop/Publications/NCCT_paper_DL/Data/labelsTr_Jeremy_200'
    path_pred = '/Users/sophieostmeier/Desktop/Publications/NCCT_paper_DL/Data/best_conf_mirror_227'
    path_NCCT = '/Users/sophieostmeier/Desktop/Publications/NCCT_paper_DL/Data/imagesTr'
    path_montages = '/Volumes/T7/NCCT_project_ncctROI/0_interrater_analysis/montages'

    path_df = None

    montage_loop([path_Ben, path_Jeremy, path_pred], path_NCCT, path_montages)