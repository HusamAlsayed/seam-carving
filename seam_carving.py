"""seam-carving.py
  Written by Husam Alsayed Ahmad 
  AI Engineer
  husamalsayedahmad@gmail.com 
  August 2021
"""
import numpy as np
import os
import cv2
import copy
from numba import jit
import argparse
from PIL import Image
import sys
import numpy as np
from scipy.ndimage.filters import convolve
parser = argparse.ArgumentParser(description = 'reduce the image with seam-carving algorithm')
parser.add_argument('-image', type = str, help = 'path of an image')
parser.add_argument('-wcut', type = int, help = 'the number of pixels to remove in the width')
parser.add_argument('-hcut', type = int, help = 'the number of pixels to remove in the height')
args = parser.parse_args()
EPS = 1e-8
def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.sqrt(convolve(img, filter_du)**2  + (convolve(img, filter_dv)**2))

    energy_map = convolved.sum(axis=2)

    return energy_map

@jit(nopython=True)
def get_least_energy_columns(dp,arr,i,j):
  if i == arr.shape[0]:
    return 0
  if j == arr.shape[1]:
    return 1e12
  if dp[i][j] != -1:
    return dp[i][j]
  ret = 1e12
  for k in range(-1,2):
    if j + k < 0 or j + k >= arr.shape[1]:
      continue
    ret = min(ret, get_least_energy_columns(dp,arr,i + 1,j + k) + arr[i,j])
  dp[i][j] = ret
  return ret

@jit(nopython=True)
def build_least_energy_columns(dp, lst,arr,i,j):
  if i == arr.shape[0]:
    return 0
  if j == arr.shape[1]:
    return 1e12
  ret = 1e12
  for k in range(-1,2):
    if j + k < 0 or j + k >= arr.shape[1]:
      continue
    ret = min(ret, get_least_energy_columns(dp, arr,i + 1,j + k) + arr[i,j])
  for k in range(-1,2):
    if j + k < 0 or j + k >= arr.shape[1]:
      continue
    av = get_least_energy_columns(dp, arr,i + 1,j + k) + arr[i,j]
    if abs(av - ret) < EPS:
      lst[i] = j + k
      build_least_energy_columns(dp, lst,arr,i + 1,j + k)
      break

@jit(nopython=True)
def get_least_energy_rows(dp,arr,i,j):
  if j == arr.shape[1]:
    return 0
  if i == arr.shape[0]:
    return 1e12
  if dp[i][j] != -1:
    return dp[i][j]
  ret = 1e12
  for k in range(-1,2):
    if i + k < 0 or i + k >= arr.shape[0]:
      continue
    ret = min(ret, get_least_energy_rows(dp,arr,i + k,j + 1) + arr[i,j])
  dp[i][j] = ret
  return ret

@jit(nopython=True)
def build_least_energy_rows(dp, lst,arr,i,j):
  if j == arr.shape[1]:
    return 0
  if i == arr.shape[0]:
    return 1e12
  ret = 1e12
  for k in range(-1,2):
    if i + k < 0 or i + k >= arr.shape[0]:
      continue
    ret = min(ret, get_least_energy_rows(dp, arr,i + k, j + 1) + arr[i,j])
  for k in range(-1,2):
    if i + k < 0 or i + k >= arr.shape[0]:
      continue
    av = get_least_energy_rows(dp, arr,i + k,j + 1) + arr[i,j]
    if abs(av - ret) < EPS:
      lst[j] = i + k
      build_least_energy_rows(dp, lst,arr,i + k,j + 1)
      break

@jit(nopython=True)
def re_initialize(image_shape):
  dp = np.zeros(shape = (image_shape[0],image_shape[1] ))
  for i in range(dp.shape[0]):
    for j in range(dp.shape[1]):
      dp[i,j] = -1
  return dp

def get_deleted_lst(sobel_img,columns = True):
  ans = 1e12 
  dp = re_initialize(sobel_img.shape)
  if columns:
    lst = np.zeros(shape = (sobel_img.shape[0],),dtype='int')
    for i in range(sobel_img.shape[1]):
      v = get_least_energy_columns(dp, sobel_img, 0, i)
      if v < ans:
        ans = v
        lst = np.zeros(shape = (sobel_img.shape[0],),dtype='int')
        build_least_energy_columns(dp, lst, sobel_img, 0, i)
  else:
    lst = np.zeros(shape = (sobel_img.shape[1],),dtype='int')
    for i in range(sobel_img.shape[0]):
      v = get_least_energy_rows(dp, sobel_img, i, 0)
      if v < ans:
        ans = v
        lst = np.zeros(shape = (sobel_img.shape[1],),dtype='int')
        build_least_energy_rows(dp, lst, sobel_img, i, 0)
  return lst

def get_new_image(original_image,num_deleted_columns,num_deleted_rows):
  for i in range(num_deleted_columns):
    sobel_image = calc_energy(original_image)
    lst = get_deleted_lst(sobel_image)
    cutted_image = np.zeros(shape = (original_image.shape[0],original_image.shape[1] - 1,original_image.shape[2]))
    for index, value in enumerate(lst):
      new_image_row =  np.concatenate( [ original_image[index,:value,:] , original_image[index, value + 1:,:]] )
      cutted_image[index,:,:] = new_image_row
    original_image = copy.deepcopy(cutted_image)
  
  for i in range(num_deleted_rows):
    sobel_image = calc_energy(original_image)
    lst = get_deleted_lst(sobel_image,False)
    cutted_image = np.zeros(shape = (original_image.shape[0] - 1,original_image.shape[1],original_image.shape[2]))
    for index, value in enumerate(lst):
      new_image_col =  np.concatenate( [ original_image[:value,index,:] , original_image[value + 1:, index,:]] )
      cutted_image[:,index,:] = new_image_col
    original_image = copy.deepcopy(cutted_image)
  return original_image


if __name__ == '__main__':
  image = cv2.imread(args.image)
  name = args.image.split('/')[-1]
  new_image = get_new_image(image,100,100)
  folder_name = './outputs'
  if not os.path.exists(folder_name):
        os.mkdir(folder_name)
  cv2.imwrite(f'{folder_name}/{name}',new_image)



