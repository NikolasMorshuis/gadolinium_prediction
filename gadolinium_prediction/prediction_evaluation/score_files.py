"""
This file provides several functions to calculate several quantities comparing the predictions with the groundtruth
"""

import numpy as np
import util.import_functions as import_functions
import nibabel as nib
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from skimage.measure import compare_ssim as ssim
from prg import prg
import pandas as pd
from sklearn.externals import joblib
import torch
import os


class Score_files():
    def __init__(self):
        pass

    def cut_off_outliers(self, image, percentile_lower=0.01, percentile_upper=99.99):
        cut_off_lower = np.percentile(image, percentile_lower)
        cut_off_upper = np.percentile(image, percentile_upper)
        image[image < cut_off_lower] = cut_off_lower
        image[image > cut_off_upper] = cut_off_upper
        return(image)

    def load_diff(self, path, patient, upper_cut):
        """
        loads the T1-Sequence of the given patient on the given path
        """

        sub_directory=str(patient)
        new_path=path+'/*0'+sub_directory
        new_file = glob.glob(new_path)

        # mask:
        mask_str='/*_mask.nii.gz'
        filename=new_file[0]+mask_str
        file_new=glob.glob(filename)
        img=nib.load(file_new[0])
        header = img.header
        affine = img.affine
        mask_data=img.get_data()
        mask_data=np.where(mask_data>=0.5,1,0)
        so=np.shape(mask_data)
        # same shape as data
        mask=np.zeros((1,1, so[0], so[1], so[2]))
        mask[0,0,:, :,:]=mask_data[:,:,:]

        filename = new_file[0] + '/*' + 'T1' + '.nii.gz'
        filen = glob.glob(filename)
        img = nib.load(filen[0])
        img_data = img.get_data()
        img_filtered = img_data * mask_data

        filename_km = new_file[0] + '/*' + 'T1_KM' + '.nii.gz'
        filen = glob.glob(filename_km)
        img = nib.load(filen[0])
        img_data = img.get_data()
        img_filtered_km = img_data * mask_data

        y = import_functions.preprocessing(path, patient)
        diff_img = img_filtered_km - img_filtered
        img_filtered = self.cut_off_outliers(diff_img, 0, upper_cut)

        diff_normalized = y.normalize_0_1(np.where(img_filtered < 0, 0, img_filtered))
        img_filtered = diff_normalized
        diff_data = np.zeros((1,1,so[0], so[1], so[2]))
        diff_data[0, 0, :, :, :] = img_filtered
        return(img_filtered, mask_data, affine, header)

    def load_prediction(self, mask, path, patient, filename_name):
        sub_directory=str(patient) # '0'
        new_path=path+'/*0'+sub_directory
        print(new_path)
        new_file = glob.glob(new_path)
        print(new_file)
        filename = new_file[0] + '/*' + filename_name + '.nii.gz'
        filen = glob.glob(filename)
        img = nib.load(filen[0])
        img_data = img.get_data()
        img_filtered = img_data * mask
        img_filtered = np.where(img_filtered < 0, 0, img_filtered)
        return(img_filtered)

    def Youdens_J(self, tf_prediction=None, tf_groundtruth=None):
        if not tf_prediction:
            tf_prediction = self.prediction
        if not tf_groundtruth:
            tf_groundtruth = self.test
        tp = np.ones(tf_prediction.shape)

        #true positives:
        tp1 = tp[tf_prediction*tf_groundtruth]
        tp_sum = np.sum(tp1)

        # false negatives
        inv_pred = ~ tf_prediction
        fn = tp[inv_pred*tf_groundtruth]
        fn_sum = np.sum(fn)

        #true negative
        inv_groundtruth = ~tf_groundtruth
        tn = tp[inv_pred*inv_groundtruth]
        tn_sum = np.sum(tn)

        #false positives
        fp = tp[tf_prediction* inv_groundtruth]
        fp_sum = np.sum(fp)

        sensitivity = tp_sum/(tp_sum+fn_sum)
        specificity = tn_sum/(tn_sum+fp_sum)

        J = sensitivity+specificity-1

        return(J, tp_sum, fp_sum, tn_sum, fn_sum)

    def complete_cycle(self, batch_prediction, batch_label, batch_mask, threshold_new=None):
        """
        Takes as input an image or array and returns the J_score, tp,fp,tn and fn
        """
        mask_tf=np.where(batch_mask>0.5,True,False)
        tf_prediction=batch_prediction[mask_tf]#batch_prediction[array]#self.prediction_brain
        tf_groundtruth=batch_label[mask_tf]#batch_label[array]#self.y_test_new
        self.tf_prediction=tf_prediction

        if threshold_new:
            self.threshold=threshold_new
            prediction_in=tf_prediction>threshold_new
        else:
            prediction_in=tf_prediction>self.threshold

        # filter contrast enhanced voxels in ground truth
        self.test=tf_groundtruth>self.threshold_orig
        self.prediction=prediction_in
        J, tp, fp, tn, fn=self.Youdens_J()
        return(J, tp, fp, tn, fn)

    def MSE_batch(self, prediction, label, mask):
        mask_tf=np.where(mask>0.5, 1,0)
        pre=np.multiply(prediction, mask_tf)
        lab=np.multiply(label, mask_tf)
        MSE=np.sum(np.square(pre-lab))/np.sum(mask_tf)
        return(MSE)

    def MAE_batch(self, prediction, label, mask):
        pre = prediction*mask
        lab = label*mask
        MSA = np.sum(np.abs(pre-lab))/np.sum(mask)
        return(MSA)

    def mutual_info(self, prediction, label, mask):
        correlation = np.corrcoef(prediction[mask==1], label[mask==1])
        hist_2d, x_edges, y_edges = np.histogram2d(
            prediction[mask == 1],
            label[mask == 1],
            bins=200)
        hist_2d_log = np.zeros(hist_2d.shape)
        non_zeros = hist_2d != 0
        hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
        mut_info=self.mutual_information(hist_2d)
        return(mut_info)

    def mutual_information(self, hgram):
        """
        Mutual information for joint histogram
        """
        # Convert bins counts to probability values
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1) # marginal for x over y
        py = np.sum(pxy, axis=0) # marginal for y over x
        px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        mut_info = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        return(mut_info)

    def precision_recall(self, prediction, gt, mask):
        threshold = filters.threshold_otsu(gt[mask==1])
        binary_gt = np.where(gt>threshold, 1, 0)[mask==1]
        prediction_b = prediction[mask == 1]
        binary_gt = np.array(binary_gt)
        prediction_b = np.array(prediction_b)
        precision, recall, thresholds = precision_recall_curve(np.ravel(binary_gt), np.ravel(prediction_b))
        auc_score = auc(recall, precision)
        return(precision,recall,thresholds, auc_score)

    def precision_recall_gain(self, prediction, gt, mask):
        threshold = filters.threshold_otsu(gt[mask==1])
        binary_gt = np.where(gt>threshold, 1, 0)[mask==1]
        prediction_b = prediction[mask == 1]
        binary_gt = np.array(binary_gt)
        prediction_b = np.array(prediction_b)
        prg_curve = prg.create_prg_curve(np.ravel(binary_gt), np.ravel(prediction_b))
        auprg = prg.calc_auprg(prg_curve)
        return(auprg)

    def ssim_score(self, prediction, gt, mask):
        n_min = np.min(np.nonzero(mask), axis=1)
        n_max = np.max(np.nonzero(mask), axis=1)
        pred_cropped = prediction[n_min[0]-1:n_max[0]+1, n_min[1]-1:n_max[1]+1,n_min[2]:n_max[2]]
        gt_cropped = gt[n_min[0]-1:n_max[0]+1, n_min[1]-1:n_max[1]+1,n_min[2]:n_max[2]]
        ssim_value = ssim(pred_cropped, gt_cropped)
        return(ssim_value)

    def roc_curve(self, prediction, gt, mask, return_tp_fp=False, plot_wanted=False):
        # create the binary image:
        threshold = filters.threshold_otsu(gt[mask==1])
        binary_gt = np.where(gt>threshold, 1, 0)[mask==1]
        prediction_b = prediction[mask==1]
        binary_gt = np.array(binary_gt)
        prediction_b = np.array(prediction_b)
        fp, tp, thresholds = roc_curve(np.ravel(binary_gt), np.ravel(prediction_b))#youd.ROC_curve(binary_gt, prediction = prediction_b)
        if plot_wanted:
            plt.plot(fp, tp)
            plt.show()
        else:
            pass
        auc_value = auc(fp, tp)
        print(auc_value)
        if return_tp_fp:
            return fp, tp, thresholds, auc_value
        else:
            return auc_value

    def reduce_len_tp_fp(self, tp, fp, n):
        fp_array = np.arange(n) / np.float(n)
        tp_array = []
        for i in fp_array:
            best_index = np.argmin(np.abs(fp - i))
            tp_array.append(tp[best_index])
        return(fp_array, tp_array)

    def psnr(self, mse):
        return(10*np.log10(1/(mse)))


    def return_all_scores(self, prediction, gt, mask, calc_perc_info=False):
        J, dice = self.positives_negatives(prediction, gt, mask)
        precision, recall, thr, auc_value_precisionrecall = self.precision_recall(prediction, gt, mask)
        auc_value_gain = self.precision_recall_gain(prediction, gt, mask)
        pred_ssim = prediction*mask
        gt_ssim = gt*mask
        ssim = self.ssim_score(pred_ssim, gt_ssim, mask)
        fp, tp, thr, auc_value = self.roc_curve(prediction, gt, mask, return_tp_fp=True)
        mutual_info = self.mutual_info(prediction, gt, mask)
        mse = self.MSE_batch(prediction, gt, mask)
        mae = self.MAE_batch(prediction, gt, mask)
        psnr = self.psnr(mse)
        if calc_perc_info:
            _, perc_info = self.perceptual_information(prediction, gt, 'alex')
        else:
            perc_info=-1
        return(mse, mae, J, dice, auc_value, mutual_info, perc_info, auc_value_precisionrecall, auc_value_gain, ssim, psnr)

    def return_scores_from_model(self, prediction_folder, data_folder, patient_list, prediction_file_name):
        scores = ['mse', 'mae', 'J', 'DICE', 'AUC_value', 'mutual_info', 'Perc_Sim', 'Prec_recall', 'PRGain', 'SSIM',
                  'PSNR', 'Patient_ID']
        df = pd.DataFrame(columns=scores)
        df_tp = pd.DataFrame(columns=np.arange(0, 100))
        for patient in patient_list:
            print(patient)
            path_to_prediction = os.path.join(prediction_folder, patient)
            path_to_data = os.path.join(data_folder, patient)
            diff_data, mask, affine, header = self.load_diff(path_to_data, 99.98)
            prediction = self.load_prediction(mask, path_to_prediction, patient, filename_name=prediction_file_name)
            mask_used = mask
            mse, mae, J, dice, auc_value, mutual_info, perc_info, prec_recall, pr_gain, ssim, psnr = \
                self.return_all_scores(prediction, diff_data, mask_used)
            df.loc[str(patient)] = [mse, mae, J, dice, auc_value, mutual_info, perc_info, prec_recall,
                                    pr_gain, ssim, psnr, patient]
            print([mse, mae, J, dice, auc_value, mutual_info, perc_info, psnr])
            fp, tp, thr, auc_value = self.roc_curve(prediction, diff_data, mask, return_tp_fp=True)
            fp_red, tp_red = self.reduce_len_tp_fp(tp, fp, 100)
            df_tp.loc[str(patient)] = tp_red
        return (df, df_tp)


if __name__ == '__main__':
    # pr = Score_files()
    # prediction_folder = # Folder containing the predictions of all patients
    # data_folder = # Folder containing the input data
    # patient_list = # List of patient_folders, for which the scores shall be calculated
    # prediction_file_name = # Name of prediction files
    # df, df_tp = pr.return_scores_from_model(prediction_folder, data_folder, patient_list, prediction_file_name)
    pass
