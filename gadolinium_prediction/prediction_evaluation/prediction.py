import numpy as np
import glob
import nibabel as nib
import os
import util.import_functions as import_functions
import torch
import torch.nn as nn
from scipy import stats
from patchcreator import *


class nii_gz_creator():
    def __init__(self):
        pass

    def cut_off_outliers(self, image, percentile_lower=0.2, percentile_upper=99.8):
        cut_off_lower = np.percentile(image, percentile_lower)
        cut_off_upper = np.percentile(image, percentile_upper)
        image[image < cut_off_lower] = cut_off_lower
        image[image > cut_off_upper] = cut_off_upper
        return(image)

    def load_t1(self, path):
        """
        loads the T1-Sequence of the given patient on the given path
        """
        new_file = glob.glob(os.path.join(path, '*'))
        print('nii files:', new_file)
        # mask:
        mask_str = '*_mask.nii.gz'
        filename = os.path.join(path, mask_str)
        file_new = glob.glob(filename)
        print('path of mask:', file_new)
        img = nib.load(file_new[0])
        header = img.header
        affine = img.affine
        mask_data = img.get_data()
        mask_data = np.where(mask_data >= 0.5, 1, 0)
        so = np.shape(mask_data)
        # same shape as data
        mask=np.zeros((1,1, so[0], so[1], so[2]))
        mask[0, 0, :, :, :] = mask_data[:, :, :]
        filename = os.path.join(path, '*_T1.nii.gz')
        print('pathbefore t1:', filename)
        filen = glob.glob(filename)
        print('path of T1:', filen)
        img = nib.load(filen[0])
        img_data = img.get_data()
        img_filtered = img_data * mask_data
        t1_data = np.zeros((1,1,so[0], so[1], so[2]))
        t1_data[0, 0, :, :, :] = img_filtered
        return(t1_data, affine, header)

    def prediction(self, path, model, bayesian=False):
        """
        Function to create predictions given a path to MRI-data from a patient and a model
        :param path:
        :param model:
        :param difference:
        :param bayesian:
        :return:
        """
        device = torch.device("cuda:0")
        if bayesian:
            for m in model.modules():
                # BatchNorm Layers are on eval mode while dropout is still active:
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                if isinstance(m, nn.Dropout):
                    m.train()
        else:
            model.eval()
        y = import_functions.preprocessing()
        _, data_matrix, label_matrix, mask, affine, header = y.import_data_nifti3d(path=path,
                                                                                    crop_img=False,
                                                                                   name_T1CE='T1_KM')
        channels = np.shape(data_matrix)[1]
        print('shape data:', np.shape(data_matrix))
        data_shape = np.shape(data_matrix)
        self.n0_min = np.zeros(channels)
        self.n0_max = np.shape(data_matrix)
        self.orig_shape = np.shape(data_matrix)

        patch_shape = np.array([1, channels, 128, 128, 128])
        indices = index_creator(data_matrix, patch_shape, 4)
        predicted_patches = np.zeros([1, 2, 128, 128, 128])
        predicted_patches = np.expand_dims(predicted_patches, 0)
        predicted_patches = np.repeat(predicted_patches, len(indices), 0)
        print(indices[1, 0])
        print(patch_shape)
        for i in range(len(indices)):
            """" How you predict the patches depends on which framework you use, which dimensionality your data has etc.
             An example is provided below on how the prediction process can look like when using pytorch. """
            patch = data_matrix[indices[i, 0]:indices[i, 0]+patch_shape[0], indices[i, 1]:indices[i, 1]+patch_shape[1], indices[i, 2]:indices[i, 2]+patch_shape[2], indices[i, 3]:indices[i, 3]+patch_shape[3], indices[i, 4]:indices[i, 4]+patch_shape[4]]
            patch_torch = torch.tensor(patch).float().to(device)
            prediction = model(patch_torch)
            prediction_numpy = prediction.cpu().detach().numpy()
            del prediction
            predicted_patches[i] = prediction_numpy
            print(i)
        matrix = aggregate_patches([1, 2, data_shape[2], data_shape[3], data_shape[4]], indices, predicted_patches)
        print('success')

        # Postprocessing of the prediction of the T1-CE and T1 difference:
        output_matrix = np.clip(matrix[:, 0:1], -1, 1)
        output_mat = y.normalize_0_1_from_min1_1(output_matrix)
        label_matrix = y.normalize_0_1_from_min1_1(label_matrix)
        output_mat = output_mat*mask

        # Postprocessing of the uncertainty prediction: Calculation of the variance of the laplace-distribution
        # given log(sigma) (log(sigma) is the second output of the model):
        aleatoric_uncertainty = 2 * np.exp(2*matrix[:, 1:2])

        torch.cuda.empty_cache()
        return(output_mat, aleatoric_uncertainty, label_matrix, mask)


    def regain_values2(self, diff_0_1, T1):
        """
        :param diff_0_1: difference normalized between 0 and 1
        :param T1_batchnormed: batchnormed T1
        :return: KM in original
        """
        diff = diff_0_1 * 1000
        print('max predicted value:', np.max(diff))
        T1_KM = T1 + diff
        return (T1_KM)

    def return_output(self, diff_0_1):
        """
        :param y: y from import function
        :param diff_0_1: difference image normalized between 0 and 1
        :return: diff adjusted for viewing
        """
        print('n0_max:', self.n0_max)
        diff_reshaped=np.zeros(self.orig_shape)
        diff_reshaped[self.n0_min[0]:self.n0_max[0], self.n0_min[1]:self.n0_max[1], self.n0_min[2]:self.n0_max[2],self.n0_min[3]:self.n0_max[3],self.n0_min[4]:self.n0_max[4]] = diff_0_1
        return(diff_reshaped)

    def regain_values_sigma(self, y, log_sigma2):
        """
        :param log_sigma2: The logarithm of sigma**2
        :return: sigma**2
        """
        orig_shape = np.zeros(y.orig_shape)
        orig_shape[y.n0_min[0]:y.n0_max[0], y.n0_min[1]:y.n0_max[1], y.n0_min[2]:y.n0_max[2], y.n0_min[3]:y.n0_max[3],
        y.n0_min[4]:y.n0_max[4]] = log_sigma2
        sigma2 = np.exp(log_sigma2)
        return (sigma2)

    def save_prediction(self, T1_KM_pred, mask, affine, header, output_directory, name):
        out_nii = np.where(mask[0, 0, :, :, :] == 1, T1_KM_pred[0, 0, :, :, :], 0)
        new_image = nib.nifti1.Nifti1Image(out_nii, affine=affine, header=header)
        file_to_be_saved = os.path.join(output_directory, name)
        try:
            nib.save(new_image, file_to_be_saved)
        except IOError:
            os.makedirs(file_to_be_saved)
            nib.save(new_image, file_to_be_saved)
        print('saved here:', file_to_be_saved)
        return 0

    def predict_patient_and_create_output(self, path, model, output_directory):
        """
        Function to predict the T1_CE image and the aleatoric uncertainty for a given patient defined by the path input.
        Note that unlike in the publication, we do not sample the output here for reasons of speed and simplicity.
        If you like to obtain an output averaged over the parameter distribution (dropout), you can set the bayesian
        parameter in the function self.prediction to True and calculate the average over the output_mat.
        You can then also calculate the epistemic uncertainty by calculating the variance over the sampled predictions.
        :param path: path to patient directory
        :param model: path to model
        :param output_directory: path to output directory
        :return: 0
        """
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)
        t1, affine, header = self.load_t1(path)
        header.set_data_dtype('float32')

        output_mat, aleatoric_uncertainty, label_mat, mask = self.prediction(path, model, bayesian=False)
        # Save the prediction
        self.save_prediction(output_mat, mask, affine, header, output_directory, 'output_1.nii.gz')
        print('prediction saved')

        # Save the aleatoric uncertainty prediction:
        self.save_prediction(aleatoric_uncertainty, mask, affine, header, output_directory, 'unc_output_1.nii.gz')
        print('uncertainty saved')

        # Save the predicted T1_CE image
        T1_CE = self.regain_values2(diff_0_1=output_mat, T1=t1)
        self.save_prediction(T1_CE, mask, affine, header, output_directory, 'm_0.nii.gz')
        print('prediction saved successfully')
        print('saved')
        return 0


if __name__=='__main__':
    print('start')
    a = nii_gz_creator()
    model_name = 'model_final.pkl'
    model_location = './data_output'
    data_path = './data/test'
    output_directory = './data_predictions'
    files = glob.glob(os.path.join(data_path, '*'))
    print(files)
    try:
        loaded_model = torch.load(os.path.join(model_location, model_name), map_location='cuda:0')
        print('model loaded')
    except FileNotFoundError:
        print('file not found error, please check the path')
        exit()
    a.predict_patient_and_create_output(files[0], loaded_model, output_directory)

