import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(orig_images_path, foggy_images_path):


	train_list = []
	val_list = []
	
	image_list_foggy = glob.glob(foggy_images_path + "*.jpg")

	tmp_dict = {}

	for image in image_list_foggy:
		image = image.split("/")[-1]
		key = image.split("_")[0] + "_" + image.split("_")[1] + ".jpg"
		if key in tmp_dict.keys():
			tmp_dict[key].append(image)
		else:
			tmp_dict[key] = []
			tmp_dict[key].append(image)


	train_keys = []
	val_keys = []

	len_keys = len(tmp_dict.keys())
	for i in range(len_keys):
		if i < len_keys*9/10:
			train_keys.append(list(tmp_dict.keys())[i])
		else:
			val_keys.append(list(tmp_dict.keys())[i])


	for key in list(tmp_dict.keys()):

		if key in train_keys:
			for foggy_image in tmp_dict[key]:

				train_list.append([orig_images_path + key, foggy_images_path + foggy_image])


		else:
			for foggy_image in tmp_dict[key]:

				val_list.append([orig_images_path + key, foggy_images_path + foggy_image])



	random.shuffle(train_list)
	random.shuffle(val_list)

	return train_list, val_list

	

class unfogging_loader(data.Dataset):

	def __init__(self, orig_images_path, foggy_images_path, mode='train'):

		self.train_list, self.val_list = populate_train_list(orig_images_path, foggy_images_path) 

		if mode == 'train':
			self.data_list = self.train_list
			print("Total training examples:", len(self.train_list))
		else:
			self.data_list = self.val_list
			print("Total validation examples:", len(self.val_list))

		

	def __getitem__(self, index):

		data_orig_path, data_foggy_path = self.data_list[index]

		data_orig = Image.open(data_orig_path)
		data_foggy = Image.open(data_foggy_path)

		data_orig = data_orig.resize((480,640), Image.ANTIALIAS)
		data_foggy = data_foggy.resize((480,640), Image.ANTIALIAS)

		data_orig = (np.asarray(data_orig)/255.0) 
		data_foggy = (np.asarray(data_foggy)/255.0) 

		data_orig = torch.from_numpy(data_orig).float()
		data_foggy = torch.from_numpy(data_foggy).float()

		return data_orig.permute(2,0,1), data_foggy.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)

