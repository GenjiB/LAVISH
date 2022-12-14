from __future__ import print_function
import argparse
from base_options import BaseOptions
from gpuinfo import GPUInfo 
import os
args = BaseOptions().parse()

mygpu = GPUInfo.get_info()[0]
gpu_source = {}

if 'N/A' in mygpu.keys():
	for info in mygpu['N/A']:
		if info in gpu_source.keys():
			gpu_source[info] +=1
		else:
			gpu_source[info] =1

for gpu_id in args.gpu:
	gpu_id = str(gpu_id)

	if gpu_id not in gpu_source.keys():
		print('go gpu:', gpu_id)
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
		break
	elif gpu_source[gpu_id] < 1:
		print('go gpu:', gpu_id)
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
		break

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.cuda.amp import autocast
import random
from dataloader import *
from nets.net_trans import MMIL_Net
from utils.eval_metrics import segment_level, event_level
import pandas as pd
from ipdb import set_trace
import wandb
from PIL import Image
from criterion import YBLoss,YBLoss2, InfoNCELoss, MaskInfoNCELoss



from torch.optim.lr_scheduler import StepLR
from einops import rearrange
from sklearn.manifold import TSNE

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = False

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)




def train(args, model, train_loader, optimizer, criterion, epoch):
	model.train()
	nceloss = InfoNCELoss(τ=args.margin1)
	mseloss = torch.nn.MSELoss()
	ybloss = YBLoss()
	ybloss2 = YBLoss2()
	
	accum_iter = 8

	rand_train_idx =11
	
	####### --------> yb: calculate mean and std
	# mean = []
	# std = []
	# for batch_idx, sample in enumerate(train_loader):
	# 	audio_spec, gt = sample['audio_spec'].to('cuda'), sample['GT'].to('cuda')

	# 	b,t,w,h = audio_spec.shape

	# 	audio_spec =  rearrange(audio_spec, 'b t w h -> (b t) (w h)')

	# 	cur_mean = torch.mean(audio_spec, dim=-1)
	# 	cur_std = torch.std(audio_spec, dim=-1)
	# 	mean.append(cur_mean)
	# 	std.append(cur_std)
	# torch.hstack(mean)
	# torch.hstack(std)
	# ######### <---------
	mean = []
	std = []
	for batch_idx, sample in enumerate(train_loader):
		

		audio_spec, gt = sample['audio_spec'].to('cuda'), sample['GT'].to('cuda')

		b,t,w,h = audio_spec.shape

		audio_spec =  rearrange(audio_spec, 'b t w h -> (b t) (w h)')

		cur_mean = torch.mean(audio_spec, dim=-1)
		cur_std = torch.std(audio_spec, dim=-1)
		mean.append(cur_mean)
		std.append(cur_std)

	

	print('mean: ',torch.hstack(mean).mean().item(),'std: ',torch.hstack(std).mean().item())
	set_trace()


def main():

	# Training settings
	# from base_options import BaseOptions
	# args = BaseOptions().parse()
	



	if args.wandb:
		wandb.init(config=args, project="ada_av",name=args.model_name)

	if args.model == 'MMIL_Net':
		model = MMIL_Net(args).to('cuda')
	else:
		raise ('not recognized')

	## -------> condition for wandb tune
	if args.start_tune_layers > args.start_fusion_layers: 
		exit()
	#### <------
	if args.mode == 'train':
		########## note for fast training #########
		# print('loading pre-training!!!!!!!!!!!!!!!!!')
		# model.load_state_dict(torch.load(args.model_save_dir + "MMIL_Net_52.35.pt"))
		########## note for fast training #########
		train_dataset = LLP_dataset(opt = args)
		val_dataset = LLP_dataset(opt = args, mode='test')
		# val_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform = transforms.Compose([
		# 									   ToTensor()]))
		
		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory = True)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory = True)

		param_group = []
		train_params = 0
		total_params = 0
		additional_params = 0
		for name, param in model.named_parameters():
			
			param.requires_grad = False
			### ---> compute params
			tmp = 1
			for num in param.shape:
				tmp *= num
			if 'my_blocks' in name or 'my_mlp_forward' in name or 'AST.v.patch_embed'in name or 'swin'in name:
				param.requires_grad = False
				
				# for num in param.shape:
				# 	tmp *= num
				total_params += tmp
				# train_params += tmp
			# ### <----



			# if  'audio_adapter_blocks' in name :  #'my_blocks' in name or 'my_mlp_forward' in name or 'adapter' in name or 'my_mlp_forward' in name 
			# 	print(name)
			# 	param.requires_grad = False
			# 	train_params += tmp
			# 	additional_params += tmp
			# 	total_params += tmp
			
			elif 'adapter_blocks' in name:
				param.requires_grad = True
				train_params += tmp
				additional_params += tmp
				total_params += tmp
				print('########### train layer:', name)
			# elif 'norm' in name:
			# 	param.requires_grad = True
			# 	train_params += tmp
				# print('########### train layer:', name)
			elif 'mlp_class' in name:
				param.requires_grad = True
				train_params += tmp
				total_params += tmp
				additional_params += tmp
	
			if 'mlp_class' in name:
				param_group.append({"params": param, "lr":args.lr_mlp})
			else:
				param_group.append({"params": param, "lr":args.lr})
		print('####### Trainable params: %0.4f  #######'%(train_params*100/total_params))
		print('####### Additional params: %0.4f  ######'%(additional_params*100/(total_params-additional_params)))
		print('####### Total params in M: %0.1f M  #######'%(total_params/1000000))
		

		optimizer = optim.Adam(param_group)
		# optimizer = optim.Adam(model.parameters(), lr=args.lr)

		# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=args.decay)
		# scheduler = StepLR(optimizer, step_size=15000, gamma=0.1)
		# criterion = nn.BCEWithLogitsLoss()
		criterion = nn.BCELoss()
		best_F = 0
		count = 0
		for epoch in range(1, args.epochs + 1):
			train(args, model, train_loader, optimizer, criterion, epoch=epoch)
			# scheduler.step()
			F_event  = eval(model, val_loader, args)
			

			count +=1

			if  F_event >= best_F:
				count = 0
				best_F = F_event
				# print('#################### save model #####################')
				# torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + "_%0.2f.pt"%(best_F))
				if args.wandb:
					wandb.log({"val-best": F_event})
			if count == args.early_stop:
				exit()
	elif args.mode == 'val':
		test_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir,
									st_dir=args.st_dir, transform=transforms.Compose([
				ToTensor()]))
		test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
		# model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
		eval(model, test_loader, args.label_val, args)
	else:
		test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,  st_dir=args.st_dir, transform = transforms.Compose([
											   ToTensor()]))
		test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
		# model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
		eval(model, test_loader, args.label_test, args)
if __name__ == '__main__':
	main()

