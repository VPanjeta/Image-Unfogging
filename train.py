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


def init(net):
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        net.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        net.weight.data.normal_(1.0, 0.02)
        net.bias.data.fill_(0)


def train(args):

	unfog_net = net.unfog_net().cuda()
	unfog_net.apply(init)

	train_dataset = dataloader.dehazing_loader(args.orig_images_path,
											 args.hazy_images_path)		
	val_dataset = dataloader.dehazing_loader(args.orig_images_path,
											 args.hazy_images_path, mode="val")		
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

	criterion = nn.MSELoss().cuda()
	optimizer = torch.optim.Adam(unfog_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	
	unfog_net.train()

	for epoch in range(args.num_epochs):
		for iteration, (img_orig, img_fog) in enumerate(train_loader):

			img_orig = img_orig.cuda()
			img_fog = img_fog.cuda()

			clean_image = unfog_net(img_fog)

			loss = criterion(clean_image, img_orig)

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(unfog_net.parameters(),args.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % args.display_iter) == 0:
				print("Loss at iteration", iteration+1, ":", loss.item())
			if ((iteration+1) % args.snapshot_iter) == 0:
				
				torch.save(unfog_net.state_dict(), args.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		

		# Validation Stage
		for iter_val, (img_orig, img_fog) in enumerate(val_loader):

			img_orig = img_orig.cuda()
			img_fog = img_fog.cuda()

			clean_image = unfog_net(img_fog)

			torchvision.utils.save_image(torch.cat((img_fog, clean_image, img_orig),0), args.sample_output_folder+str(iter_val+1)+".jpg")

		torch.save(unfog_net.state_dict(), args.snapshots_folder + "net.pth") 


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--orig_images_path', type=str, default="data/images/")
	parser.add_argument('--foggy_images_path', type=str, default="data/data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=10)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=8)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=200)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--sample_output_folder', type=str, default="samples/")

	args = parser.parse_args()

	if not os.path.exists(args.snapshots_folder):
		os.mkdir(args.snapshots_folder)
	if not os.path.exists(args.sample_output_folder):
		os.mkdir(args.sample_output_folder)

	train(args)








	
