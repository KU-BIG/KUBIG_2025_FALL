#Written by Dominik Waibel
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os
import torch
from skimage.transform import resize, rotate
import random
from skimage.io import imread, imsave
from skimage.transform import resize


def import_image(path_name):
    '''
    This function loads the image from the specified path
    NOTE: The alpha channel is removed (if existing) for consistency
    Args:
        path_name (str): path to image file
    return:
        image_data: numpy array containing the image data in at the given path.
    '''
    if path_name.endswith('.npy'):
        image_data = np.array(np.load(path_name))
    else:
        image_data = imread(path_name)
        # If has an alpha channel, remove it for consistency
    if np.array(np.shape(image_data))[-1] == 4:
        image_data = image_data[: ,: ,0:3]
    return image_data

def augmentation(obj, img):
    if random.choice([True, False, False]) == True:
        obj = np.flip(obj, len(np.shape(obj)) - 2).copy()
        img = np.flip(img, len(np.shape(img)) - 2).copy()
    if random.choice([True, False, False]) == True:
        obj = np.flip(obj, len(np.shape(obj)) - 3).copy()
        img = np.flip(img, len(np.shape(img)) - 3).copy()
    return obj, img




"""
The data generator will open the 3D segmentation, 2D masks and 2D images for each fold from the directory given the filenames and return a tensor
The 2D mask and the 2D image will be multiplied pixel-wise to remove the background
"""

def extract_base_name(filename):
  """
  Extracts the base name from the file name, removing the identifier part and extension.

  Args:
    filename: The input file name (e.g., 'Neu1_cell1_I2D64_t000.tif', 'Neu2_cell1_cell2_cell3_I2D64_t000.tif').

  Returns:
    The extracted base name (e.g., 'Neu1_cell1_t000', 'Neu2_cell1_cell2_cell3_t000').
  """
  parts = filename.split('_')
  identifier_index = -1
  for i, part in enumerate(parts):
      if part.startswith(('I', 'S', 'V')) and part.endswith('D64'):
          identifier_index = i
          break

  if identifier_index != -1 and len(parts) > identifier_index + 1:
    base_name_parts = parts[:identifier_index] + parts[identifier_index+1:]
    base_name = '_'.join(base_name_parts)
    base_name = os.path.splitext(base_name)[0]
    return base_name
  return None


class SHAPRDataset(Dataset):
    def __init__(self, path, test_flag=True):
        self.test_flag = test_flag
        self.path = path
        # added
        self.image_dir = os.path.join(path, 'image')
        self.mask_dir = os.path.join(path, 'mask')
        self.obj_dir = os.path.join(path, 'obj')

        self.samples = []
        base_names = {}

        # List files and group by base name
        for filename in os.listdir(self.image_dir):
            base_name = extract_base_name(filename)
            if base_name:
                if base_name not in base_names:
                    base_names[base_name] = {}
                base_names[base_name]['image'] = os.path.join(self.image_dir, filename)

        for filename in os.listdir(self.mask_dir):
             base_name = extract_base_name(filename)
             if base_name:
                if base_name not in base_names:
                    base_names[base_name] = {}
                base_names[base_name]['mask'] = os.path.join(self.mask_dir, filename)

        for filename in os.listdir(self.obj_dir):
             base_name = extract_base_name(filename)
             if base_name:
                if base_name not in base_names:
                    base_names[base_name] = {}
                base_names[base_name]['obj'] = os.path.join(self.obj_dir, filename)


        # Create a list of samples where each sample contains paths to image, mask, and obj
        for base_name, files in base_names.items():
            if 'image' in files and 'mask' in files and 'obj' in files:
                self.samples.append({
                    'base_name': base_name,
                    'image': files['image'],
                    'mask': files['mask'],
                    'obj': files['obj']
                })
            else:
                print(f"Warning: Skipping incomplete sample for base name: {base_name}")

        #if self.test_flag==True:
        #    self.path = path + "/test/"
        #else:
        #    self.path = path + "/train/"
        self.filenames = os.listdir(self.path + "/obj/")

    def __len__(self):
        return len(self.filenames)
    

    def __getitem__(self, idx):
      # Use the pre-computed sample paths from __init__
      sample_paths = self.samples[idx]

      image_dir = sample_paths['image']
      mask_dir = sample_paths['mask']
      obj_dir = sample_paths['obj']

      # Load your image, mask, and obj data using these paths
      img = import_image(mask_dir) / 255.  # Use mask_path
      bf = import_image(image_dir) / 255.  # Use image_path

      msk_bf = np.zeros((2, int(np.shape(img)[0]), int(np.shape(img)[1])))
      msk_bf[0, :, :] = img
      msk_bf[1, :, :] = bf * img
      mask_bf = msk_bf[:, np.newaxis, ...]
      e = np.concatenate((mask_bf, mask_bf), axis=1)
      e = np.concatenate((e, e), axis=1)
      e = np.concatenate((e, e), axis=1)
      e = np.concatenate((e, e), axis=1)
      e = np.concatenate((e, e), axis=1)
      mask_bf = np.concatenate((e, e), axis=1)

      if self.test_flag:
          # Assuming self.filenames is replaced by sample_paths['base_name'] or similar if needed elsewhere
          return torch.from_numpy(mask_bf).float(), sample_paths['base_name'] # Or whatever identifier you need
      else:
          obj = import_image(obj_dir) / 255. # Use obj_path
          obj = obj[np.newaxis, :, :, :]
          #obj, mask_bf = augmentation(obj, mask_bf) # Keep your augmentation if needed
          return torch.from_numpy(mask_bf).float(), torch.from_numpy(obj).float()

    # def __getitem__(self, idx):
    #     img = import_image(os.path.join(self.path, "mask", self.filenames[idx])) / 255.
    #     bf = import_image(os.path.join(self.path, "image", self.filenames[idx])) / 255.
    #     msk_bf = np.zeros((2, int(np.shape(img)[0]), int(np.shape(img)[1])))
    #     msk_bf[0, :, :] = img
    #     msk_bf[1, :, :] = bf * img
    #     mask_bf = msk_bf[:, np.newaxis, ...]
    #     e = np.concatenate((mask_bf, mask_bf), axis=1)
    #     e = np.concatenate((e, e), axis=1)
    #     e = np.concatenate((e, e), axis=1)
    #     e = np.concatenate((e, e), axis=1)
    #     e = np.concatenate((e, e), axis=1)
    #     mask_bf = np.concatenate((e, e), axis=1)
    #     if self.test_flag:
    #         return torch.from_numpy(mask_bf).float(), self.filenames[idx]
    #     else:
    #         obj = import_image(os.path.join(self.path, "obj", self.filenames[idx])) / 255.
    #         obj = obj[np.newaxis, :, :, :]
    #         #obj, mask_bf = augmentation(obj, mask_bf)
    #         return torch.from_numpy(mask_bf).float(), torch.from_numpy(obj).float()
