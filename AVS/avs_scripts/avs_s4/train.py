import os
import time
import random
import shutil
import torch
import numpy as np
import argparse

from gpuinfo import GPUInfo 
from base_options import BaseOptions
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
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
import logging

from config import cfg
from dataloader import S4Dataset
from torchvggish import vggish
from loss import IouSemanticAwareLoss

from utils import pyutils
from utils.utility import logger, mask_iou
from utils.system import setup_logging
from ipdb import set_trace

import certifi
import sys

import wandb

os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), certifi.where())



class audio_extractor(torch.nn.Module):
	def __init__(self, cfg, device):
		super(audio_extractor, self).__init__()
		self.audio_backbone = vggish.VGGish(cfg, device)

	def forward(self, audio):
		audio_fea = self.audio_backbone(audio)
		return audio_fea


if __name__ == "__main__":
	# parser = argparse.ArgumentParser()    

	# args = parser.parse_args()

	
	if args.wandb:

		wandb.init(config=args, project="ada_av_segmentation", name=args.model_name)

	if (args.visual_backbone).lower() == "resnet":
		from model import ResNet_AVSModel as AVSModel
		print('==> Use ResNet50 as the visual backbone...')
	elif (args.visual_backbone).lower() == "pvt":
		from model import PVT_AVSModel as AVSModel
		print('==> Use pvt-v2 as the visual backbone...')
	else:
		raise NotImplementedError("only support the resnet50 and pvt-v2")


	# Fix seed
	FixSeed = 123
	random.seed(FixSeed)
	np.random.seed(FixSeed)
	torch.manual_seed(FixSeed)
	torch.cuda.manual_seed(FixSeed)

	# Log directory
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir, exist_ok=True)
	# Logs
	prefix = args.session_name
	log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
	args.log_dir = log_dir

	# Save scripts
	script_path = os.path.join(log_dir, 'scripts')
	if not os.path.exists(script_path):
		os.makedirs(script_path, exist_ok=True)

	scripts_to_save = ['train.sh', 'train.py', 'test.sh', 'test.py', 'config.py', 'dataloader.py', './model/ResNet_AVSModel.py', './model/PVT_AVSModel.py', 'loss.py']
	for script in scripts_to_save:
		dst_path = os.path.join(script_path, script)
		try:
			shutil.copy(script, dst_path)
		except IOError:
			os.makedirs(os.path.dirname(dst_path), exist_ok=True)
			shutil.copy(script, dst_path)

	# Checkpoints directory
	checkpoint_dir = os.path.join(log_dir, 'checkpoints')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir, exist_ok=True)
	args.checkpoint_dir = checkpoint_dir

	# Set logger
	log_path = os.path.join(log_dir, 'log')
	if not os.path.exists(log_path):
		os.makedirs(log_path, exist_ok=True)

	setup_logging(filename=os.path.join(log_path, 'log.txt'))
	logger = logging.getLogger(__name__)
	logger.info('==> Config: {}'.format(cfg))
	logger.info('==> Arguments: {}'.format(args))
	logger.info('==> Experiment: {}'.format(args.session_name))

	# Model
	model = AVSModel.Pred_endecoder(channel=256, \
										opt=args, \
										config=cfg, \
										tpavi_stages=args.tpavi_stages, \
										tpavi_vv_flag=args.tpavi_vv_flag, \
										tpavi_va_flag=args.tpavi_va_flag)
	model = torch.nn.DataParallel(model).cuda()
	model.train()

	total_params = 0
	train_params = 0
	additional_params = 0
	for name, param in model.named_parameters():
	
		param.requires_grad = True
		### ---> compute params
		tmp = 1
		for num in param.shape:
			tmp *= num
		# if 'encoder_backbone' not in name:
		total_params += tmp
	
		if 'ViT'in name or 'swin' in name:
			param.requires_grad = False
		elif 'adapter' in name:
			additional_params += tmp
			train_params += tmp
		else:
			train_params += tmp

	
	print('####### Trainable params: %0.4f  #######'%(train_params*100/total_params))
	print('####### Additional params: %0.4f  ######'%(additional_params*100/(total_params-additional_params)))
	print('####### Total params in M: %0.1f M  #######'%(total_params/1000000))


	# for k, v in model.named_parameters():
	#         print(k, v.requires_grad)

	# video backbone
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	audio_backbone = audio_extractor(cfg, device)

	# yb:add
	# audio_backbone = torch.nn.DataParallel(audio_backbone).cuda() 
	audio_backbone.cuda()
	audio_backbone.eval()

	total_params_audio = 0
	for name, param in audio_backbone.named_parameters():
	
		tmp = 1
		for num in param.shape:
			tmp *= num

		total_params_audio += tmp
	

	
	print('####### Total audio params in M: %0.1f M  #######'%(total_params_audio/1000000))

	# Data
	train_dataset = S4Dataset('train', args)
	train_dataloader = torch.utils.data.DataLoader(train_dataset,
														batch_size=args.train_batch_size,
														shuffle=True,
														num_workers=args.num_workers,
														pin_memory=True)
	max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

	val_dataset = S4Dataset('test',args)
	val_dataloader = torch.utils.data.DataLoader(val_dataset,
														batch_size=args.val_batch_size,
														shuffle=False,
														num_workers=args.num_workers,
														pin_memory=True)

	# Optimizer
	model_params = model.parameters()
	optimizer = torch.optim.Adam(model_params, args.lr)


	avg_meter_total_loss = pyutils.AverageMeter('total_loss')
	avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')
	avg_meter_sa_loss = pyutils.AverageMeter('sa_loss')
	avg_meter_miou = pyutils.AverageMeter('miou')

	# Train
	best_epoch = 0
	global_step = 0
	miou_list = []
	max_miou = 0
	for epoch in range(args.max_epoches):
		for n_iter, batch_data in enumerate(train_dataloader):
			# imgs, audio, mask = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
			imgs, audio_spec, audio, mask = batch_data

			imgs = imgs.cuda()
			audio = audio.cuda()
			mask = mask.cuda()
			B, frame, C, H, W = imgs.shape
			# imgs = imgs.view(B*frame, C, H, W)
			mask = mask.view(B, H, W)

			audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4]) # [B*T, 1, 96, 64]

			
			# with torch.no_grad():
			#     audio_feature = audio_backbone(audio) # [B*T, 128]
			# # # # ------> yb: add
			# audio_feature = audio_feature.view(B,5, audio_feature.size(-1))
			# B x time x 128
			# <------

			output, visual_map_list, a_fea_list = model(imgs, audio_spec) # [bs*5, 1, 224, 224]
			loss, loss_dict = IouSemanticAwareLoss(output, mask.unsqueeze(1).unsqueeze(1), \
												a_fea_list, visual_map_list, \
												lambda_1=args.lambda_1, \
												count_stages=args.sa_loss_stages, \
												sa_loss_flag=args.sa_loss_flag, \
												mask_pooling_type=args.mask_pooling_type)


			avg_meter_total_loss.add({'total_loss': loss.item()})
			avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
			avg_meter_sa_loss.add({'sa_loss': loss_dict['sa_loss']})

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			global_step += 1

			if (global_step-1) % 50 == 0:
				train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, lambda_1:%.4f, lr: %.6f'%(
							global_step-1, 
							max_step, 
							avg_meter_total_loss.pop('total_loss'), 
							avg_meter_iou_loss.pop('iou_loss'), 
							avg_meter_sa_loss.pop('sa_loss'), 
							args.lambda_1, 
							optimizer.param_groups[0]['lr'])

	
				logger.info(train_log)


		# Validation:
		count = 0 
		model.eval()
		with torch.no_grad():
			for n_iter, batch_data in enumerate(val_dataloader):
				# imgs, audio, mask, _, _ = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]
				imgs, audio_spec, audio, mask,_,_ = batch_data
  

				imgs = imgs.cuda()
				audio = audio.cuda()
				mask = mask.cuda()
				B, frame, C, H, W = imgs.shape
				# imgs = imgs.view(B*frame, C, H, W)
				mask = mask.view(B*frame, H, W)
				# audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
				# with torch.no_grad():
				#     audio_feature = audio_backbone(audio)

				# # ------> yb: add
				# audio_feature = audio_feature.view(B,5, audio_feature.size(-1))
				# # <------

				output, _, _ = model(imgs, audio_spec) # [bs*5, 1, 224, 224]


				miou = mask_iou(output.squeeze(1), mask)
				avg_meter_miou.add({'miou': miou})

			miou = (avg_meter_miou.pop('miou'))
			count = count +1
			if miou > max_miou:
				model_save_path = os.path.join(checkpoint_dir, '%s_best.pth'%(args.session_name))
				torch.save(model.module.state_dict(), model_save_path)
				best_epoch = epoch
				logger.info('save best model to %s'%model_save_path)
				count = 0

				if args.wandb:
					wandb.log({"val-best": miou})
			if count == args.early_stop:
				exit()

			miou_list.append(miou)
			max_miou = max(miou_list)

			val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(epoch, miou, max_miou)
			# print(val_log)
			logger.info(val_log)

		model.train()
	logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))











