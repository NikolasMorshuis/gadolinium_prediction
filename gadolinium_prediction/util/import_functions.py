from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import nibabel as nib
import pandas as pd
import torch
import yaml

# batchgenerator elements that are used:
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.transforms.utility_transforms import NumpyToTensor

# batchgenerator augmentation elements that can be used for further data augmentation:
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.abstract_transforms import RndTransform
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform


with open("config.yaml", 'r') as stream:
    sequences_list_all = yaml.safe_load(stream)


class DataLoader3d(SlimDataLoaderBase):
    def __init__(self, data, n_channels, shape_data, BATCH_SIZE=2):
        super(DataLoader3d, self).__init__(data, BATCH_SIZE)
        self.BATCH_SIZE = BATCH_SIZE
        self.n_channels = n_channels
        if shape_data is not False:
            self.shape_data = shape_data
        else:
            self.shape_data = (96, 96, 96)

    def load_train_case(self, data_name):
        input_name = os.path.join(data_name, 'data.npy')
        gt_name = os.path.join(data_name, 'gt.npy')
        data = np.load(input_name, mmap_mode='r')
        label = np.load(gt_name, mmap_mode='r')
        return data, label

    def generate_train_batch(self):
        # data must be a list of len(num_patients), each entry in the list is a filename /or some filenames that point
        # to the files that need to be loaded
        idx = np.random.choice(len(self._data), self.BATCH_SIZE, replace=True)
        data = np.zeros((self.BATCH_SIZE, self.n_channels, self.shape_data[0], self.shape_data[1], self.shape_data[2]),
                        dtype=np.float32)
        seg = np.zeros((self.BATCH_SIZE, 2, self.shape_data[0], self.shape_data[1], self.shape_data[2]),
                       dtype=np.float32)
        for i, b in enumerate(idx):
            img, s = self.load_train_case(self._data[b])
            img, s = crop(data=img, seg=s, crop_size=self.shape_data, margins=(0, 0, 0), crop_type='random')
            data[i] = img[0]
            seg[i] = s[0]
        return {'data': data, 'seg': seg}


class preprocessing:
    def __init__(self):
        pass

    def import_data_nifti3d(self, path, crop_img=True, difference=True, name_T1='T1', name_T1CE='T1_CE'):
        new_file = glob.glob(path)

        # import the mask:
        mask_str = '/*_mask.nii.gz'
        filename = new_file[0]+mask_str
        file_new = glob.glob(filename)
        img = nib.load(file_new[0])
        mask_data = img.get_data()
        mask_data = np.where(mask_data >= 0.5, 1, 0)
        so = np.shape(mask_data)

        # same shape as data
        mask = np.zeros((1,1, so[0], so[1], so[2]))
        mask[0, 0, :, :, :] = mask_data[:, :, :]

        # Get the shape of cropped image
        if crop_img:
            n0_min,n0_max=self.crop_image_3d_spec_shape(mask)
            self.n0_min = n0_min
            self.n0_max = n0_max
            self.orig_shape = np.shape(mask)
        else:
            n0_min = (0, 0, 0, 0, 0)
            n0_max = np.shape(mask)
        self.n_batches = n0_max[3]-n0_min[3]

        # Create the data matrix
        data_mat = np.zeros((1, len(sequences_list_all), so[0], so[1], so[2]))

        # Load the sequences
        for c, s in enumerate(sequences_list_all):
            filename = os.path.join(new_file[0], '*{}.nii.gz'.format(s))
            print(new_file)
            filen = glob.glob(filename)
            print(filen)
            img = nib.load(filen[0])
            img_data = img.get_data()

            # extract the brain:
            img_filtered=img_data*mask_data

            # Save values for T1 and T1_CE to calculate the difference:
            if s == name_T1:
                T1_img = img_filtered
            elif s == name_T1CE:
                T1_CE_img = img_filtered

            # Cut off outliers (otherwise InstanceNorm might not produce nicely distributed results):
            img_filtered = self.cut_off_outliers(img_filtered, mask_data, percentile_lower=0.02, percentile_upper=99.95)

            # Normalize the data
            # InstanceNorm -> Cut off Outliers -> Normalize data to the range [0,1] -> Set data outside mask to 0
            img_data_normalized = self.instancenorm(img_filtered, mask_data)
            img_data_normalized = self.cut_off_outliers(img_data_normalized, percentile_lower=0.5, percentile_upper=99.5)
            img_data_normalized = self.normalize_0_1(img_data_normalized)
            img_data_normalized = np.where(mask_data == 1, img_data_normalized, 0)
            header = img.header
            affine = img.affine

            # build the data matrix
            data_mat[0, c, :, :, :] = img_data_normalized

        # Create diff image
        diff_img = T1_CE_img - T1_img # Calculate difference T1_CE - T1
        diff_img = self.cut_off_outliers(diff_img, mask_data, percentile_lower=0, percentile_upper=99.98)
        self.max_diff_img = np.max(diff_img)
        self.min_diff_img = np.min(diff_img)
        diff_normalized = self.normalize_0_1(np.where(diff_img<0,0,diff_img))
        self.diff_matrix = np.zeros((1, 1, so[0],so[1], so[2]))
        self.diff_matrix[0, 0, :, :, :] = diff_normalized

        # Crop the images
        self.diff_matrix = self.diff_matrix[:, :, n0_min[2]:n0_max[2], n0_min[3]:n0_max[3], n0_min[4]:n0_max[4]]
        data_mat = data_mat[:, :, n0_min[2]:n0_max[2], n0_min[3]:n0_max[3], n0_min[4]:n0_max[4]]
        mask = mask[:, :, n0_min[2]:n0_max[2], n0_min[3]:n0_max[3], n0_min[4]:n0_max[4]]
        self.T1_var = np.var(data_mat[:, 0, :, :, :])
        self.T1_mean = np.mean(data_mat[:, 0, :, :, :])
        # Normalization and Label assignment
        # Label normalization
        if difference:
            self.mean_label = np.mean(self.diff_matrix)
            self.var_label = np.var(self.diff_matrix)
            label_matrix = self.normalize_min1_1(self.diff_matrix)
        else:
            self.mean_label = np.mean(data_mat[0, 10, :, :, :])
            self.var_label = np.var(data_mat[0, 10, :, :, :])
            label_matrix = np.zeros_like(self.diff_matrix)
            label_matrix[0, 0, :, :, :] = self.normalize_min1_1(data_mat[0,10,:,:,:])

        # Prepare the data for the output
        y = [[[] for i in range(3)] for i in range(1)]
        y[0] = [data_mat[0, : -1, :, :, :], label_matrix[0, :, :, :, :], mask[0, :, :, :, :]]
        return y, data_mat[:, :-1, :, :, :], label_matrix[:, :, :, :, :], mask[:, :, :, :, :], affine, header


    def crop_image(self, mask):
        nonzeros_min=np.min(np.nonzero(mask),axis=1)
        nonzeros_max=np.max(np.nonzero(mask),axis=1)
        return(nonzeros_min, nonzeros_max)

    
    def crop_image_3d_spec_shape(self,mask):
        shape=np.shape(mask)
        nonzeros_min = np.min(np.nonzero(mask), axis=1)
        nonzeros_max = np.max(np.nonzero(mask), axis=1)
        nonzeros_min = np.maximum(np.zeros(5), nonzeros_min-10)
        nonzeros_max = np.minimum(shape, nonzeros_max+10)
        cropped_size = nonzeros_max-nonzeros_min

        # make sure that each dimension is larger than the patch-size
        for i in range(3):
            if cropped_size[2+i]<128:
                nonzeros_min[2+i] = np.maximum(0, nonzeros_min[2+i]-48)
                if nonzeros_min[2+i] - 48 < 0:
                    additional = 48 - nonzeros_min[2+i]
                else:
                    additional = 0
                nonzeros_max[2+i] = np.minimum(shape[2+i], nonzeros_max[2+i]+48+additional)
        nonzeros_max[0:2] = [1, 1]
        return nonzeros_min.astype(int), nonzeros_max.astype(int)


    def cut_off_outliers(self, image, mask=None, percentile_lower=0.02, percentile_upper=99.95):
        if mask is not None:
            cut_off_lower = np.percentile(image[mask==1], percentile_lower)
            cut_off_upper = np.percentile(image[mask==1], percentile_upper)
        else:
            cut_off_lower = np.percentile(image, percentile_lower)
            cut_off_upper = np.percentile(image, percentile_upper)
        image[image < cut_off_lower] = cut_off_lower
        image[image > cut_off_upper] = cut_off_upper
        return(image)

    def load_nifti(fname, dtype=np.float32):
        # import nibabel as nib
        nifti_obj = nib.load(fname)
        img = nifti_obj.get_data().astype(dtype)
        affine = nifti_obj.get_affine()
        header = nifti_obj.get_header()
        return img, affine, header


    def simple_normalization(self,image):
        #Normalize a 4D image
        eps=0.0000000000001
        img=image
        mini=img.min(axis=(2, 3), keepdims=True)
        maxi=img.max(axis=(2, 3), keepdims=True)
        img_new=(img-mini)/(maxi-mini+eps)
        return img_new

    def normalize_0_1(self, image):
        maxi=np.max(image)
        mini=np.min(image)
        if maxi-mini != 0:
            image_normalized=(image-mini)/(maxi-mini)
        else: # zero image
            image_normalized = image
        return(image_normalized)

    def normalize_0_1_from_min1_1(self, image):
        maxi=1
        mini=-1
        image_normalized=(image-mini)/(maxi-mini)
        return(image_normalized)


    def normalize_min1_1(self,image):
        maxi=np.max(image)
        mini=np.min(image)
        image_normalized=2*(image-mini)/(maxi-mini)-np.ones_like(image)
        return(image_normalized)

    def normalize_0_1_from_a_b(self, image, borders):
        maxi=borders[0]
        mini=borders[1]
        image_normalized=(image-mini)/(maxi-mini)
        image_normalized = np.where(image_normalized<0, 0, image_normalized)
        image_normalized = np.where(image_normalized>1, 1, image_normalized)
        return(image_normalized)

    def instancenorm(self, image, mask=None):
        eps=1e-8
        if mask is not None:
            image_array = np.ravel(image[mask==1])
        exp_in=np.mean(image_array)
        var_in=np.var(image_array)
        image=(image-exp_in)/(np.sqrt(var_in)+eps)
        return(image)

    def instancenorm_tensors(self, image):
        eps = 1e-8
        exp_in = image.mean()
        var_in = image.var()
        image = (image-exp_in)/(torch.sqrt(var_in)+eps)
        return(image)

    def instancenorm_to_0_1(self, image):
        T1_diff_re=image*np.sqrt(self.var_label)+self.mean_label
        return(T1_diff_re)

    def instancenorm_to_origsize(self, image):
        T1_diff_re=self.instancenorm_to_0_1(image)
        diff1 = T1_diff_re*(self.max_diff_img-self.min_diff_img)+self.min_diff_img
        return(diff1)

    def instancenorm_clip_normalize(self, image, mask, clip=[10, -5]):
        image=self.instancenorm(image)
        image[image>clip[0]] = clip[0]
        image[image<clip[1]]= clip[1]
        image= self.normalize_0_1(image)
        image = np.where(mask<0.5, 0, image)
        return(image)

    def instancenorm_clip_normalize_min1_1(self, image, mask, clip=[10, -5]):
        image=self.instancenorm(image)
        image[image>clip[0]] = clip[0]
        image[image<clip[1]]= clip[1]
        image= self.normalize_min1_1(image)
        image = np.where(mask<0.5, -1, image)
        return(image)


class augmentation3d:
    def __init__(self, data, n_channels, BATCH_SIZE=4, spatial_transform=True, shape_data=None):
        self.batchgen = DataLoader3d(data, n_channels=n_channels, shape_data=shape_data, BATCH_SIZE=BATCH_SIZE)
        self.batch = next(self.batchgen)
        self.intdata = self.batch
        self.spatial_transform = spatial_transform
        if shape_data is not None:
            self.shape_data = shape_data
        else:
            self.shape_data = (96, 96, 96)


    def plot_batch(self):
        batch=self.intdata
        batch_size = batch['data'].shape[0]
        plt.figure(figsize=(16, 10))
        for i in range(batch_size):
            plt.subplot(1, batch_size, i+1)
            plt.imshow(batch['data'][i, 0], cmap="gray") # only grayscale image here
        plt.show()

    def transform(self, cores=1, use_cuda=True):
        my_transforms = []
        mirror_transform = MirrorTransform(axes=(0, 1, 2))
        my_transforms.append(mirror_transform)
        if self.spatial_transform:
            spatial_transform = SpatialTransform(self.shape_data, np.array(self.intdata['data'][0,0].shape) // 2,
                     do_elastic_deform=False, alpha=(0., 1500.), sigma=(30., 50.),
                     do_rotation=True, angle_z=(0, 2 * np.pi),
                     do_scale=True, scale=(0.8, 1.2),
                     border_mode_data='constant', border_cval_data=0, order_data=1,
                     random_crop=False, order_seg=0, p_el_per_sample=0.3, p_rot_per_sample=0.3, p_scale_per_sample=0.3)
            my_transforms.append(spatial_transform)
        np_to_tensor = NumpyToTensor(cast_to='float')
        my_transforms.append(np_to_tensor)
        all_transforms = Compose(my_transforms)
        if use_cuda:
            pin_memory = True
        else:  # pin memory can not be used if cuda is not enabled (in this case).
            pin_memory = False
        multithreaded_generator = MultiThreadedAugmenter(self.batchgen, all_transforms, cores, 2, seeds=None,
                                                         pin_memory=pin_memory)
        return(multithreaded_generator)



