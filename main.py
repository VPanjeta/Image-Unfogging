import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob


def unfog_image(image_path):

	data_foggy = Image.open(image_path)
	data_foggy = (np.asarray(data_foggy) / 255.0)

	data_foggy = torch.from_numpy(data_foggy).float()
	data_foggy = data_foggy.permute(2, 0, 1)
	data_foggy = data_foggy.cuda().unsqueeze(0)

	unfog_net = net.unfog_net().cuda()
	unfog_net.load_state_dict(torch.load('snapshots/net.pth'))

	clean_image = unfog_net(data_foggy)
	torchvision.utils.save_image(torch.cat((data_foggy, clean_image),0), "results/" + image_path.split("/")[-1])
	

if __name__ == '__main__':

	test_list = glob.glob("test_images/*")

	for image in test_list:

		unfog_image(image)
		print(image, "done!")
