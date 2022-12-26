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
# from nets.net_trans_805 import MMIL_Net
from utils.eval_metrics import segment_level, event_level
import pandas as pd
from ipdb import set_trace
import wandb
from PIL import Image
from criterion import YBLoss,YBLoss2, InfoNCELoss, MaskInfoNCELoss



from torch.optim.lr_scheduler import StepLR
from einops import rearrange

import certifi
import sys
from torch.cuda import amp

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), certifi.where())


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

	scaler = amp.GradScaler()
	rand_train_idx = 11
	

	for batch_idx, sample in enumerate(train_loader):
		

		audio_spec, gt = sample['audio_spec'].to('cuda'), sample['GT'].to('cuda')
		image = sample['image'].to('cuda')

		# audio_vgg = sample['audio_vgg'].float().to('cuda')
	
		optimizer.zero_grad()

		output = model([audio_spec], image, rand_train_idx=rand_train_idx, stage='train')


		loss = criterion(output.squeeze(1),rearrange(gt, 'b t class -> (b t) class')) 

		loss.backward()
		optimizer.step()

		# weights update
		if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
			optimizer.step()

		if batch_idx % 50 == 0:
			# print(model.fusion_weight['blocks-0-attn-qkv-weight'])

			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_total: {:.6f}'.format(
				epoch, batch_idx * len(audio_spec), len(train_loader.dataset),
					   100. * batch_idx / len(train_loader), loss.item())) #

def eval(model, val_loader, args):

	model.eval()
	

	total_acc = 0
	total_num = 0

	for batch_idx, sample in enumerate(val_loader):
		

		audio_spec, gt= sample['audio_spec'].to('cuda'), sample['GT'].to('cuda')
		image = sample['image'].to('cuda')
		audio_vgg = sample['audio_vgg'].float().to('cuda')

		with torch.no_grad():
			output = model([audio_spec, audio_vgg], image)

		total_acc += (output.squeeze(1).argmax(dim=-1) == rearrange(gt, 'b t class -> (b t) class').argmax(dim=-1)).sum()
		total_num += output.size(0)


	acc = total_acc/total_num


	print('val acc: %.2f'%(acc*100))
	return acc

def main():

	# Training settings
	# from base_options import BaseOptions
	# args = BaseOptions().parse()
	



	if args.wandb:
		wandb.init(config=args, project="ada_av",name=args.model_name)


	model = MMIL_Net(args).to('cuda')


	## -------> condition for wandb tune
	if args.start_tune_layers > args.start_fusion_layers: 
		exit()
	#### <------
	if args.mode == 'train':

		train_dataset = AVE_dataset(opt = args)
		val_dataset = AVE_dataset(opt = args, mode='test')

		
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

			if 'ViT'in name or 'swin' in name:
				if 'norm' in name and args.is_vit_ln:
					param.requires_grad = bool(args.is_vit_ln)
					total_params += tmp
					train_params += tmp
				else:
					param.requires_grad = False
					total_params += tmp
				
			# ### <----

			elif 'adapter_blocks' in name:
				param.requires_grad = True
				train_params += tmp
				additional_params += tmp
				total_params += tmp
				print('########### train layer:', name, param.shape , tmp)
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

		criterion = nn.BCELoss()

		best_F = 0
		count = 0
		for epoch in range(1, args.epochs + 1):
			train(args, model, train_loader, optimizer, criterion, epoch=epoch)

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
		test_dataset = AVE_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir,
									st_dir=args.st_dir, transform=transforms.Compose([
				ToTensor()]))
		test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
		# model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
		eval(model, test_loader, args.label_val, args)
	else:
		test_dataset = AVE_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,  st_dir=args.st_dir, transform = transforms.Compose([
											   ToTensor()]))
		test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
		# model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
		eval(model, test_loader, args.label_test, args)
if __name__ == '__main__':
	main()

