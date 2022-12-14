#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
# from util import util
import torch

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument("--session_name", default="S4", type=str, help="the S4 setting")
		self.parser.add_argument("--visual_backbone", default="resnet", type=str, help="use resnet50 or pvt-v2 as the visual backbone")

		self.parser.add_argument("--train_batch_size", default=4, type=int)
		self.parser.add_argument("--val_batch_size", default=1, type=int)
		self.parser.add_argument("--max_epoches", default=15, type=int)
		self.parser.add_argument("--lr", default=0.0001, type=float)
		
		self.parser.add_argument("--wt_dec", default=5e-4, type=float)


		self.parser.add_argument('--sa_loss_flag', action='store_true', default=False, help='additional loss for last four frames')
		self.parser.add_argument("--lambda_1", default=0, type=float, help='weight for balancing l4 loss')
		self.parser.add_argument("--sa_loss_stages", default=[], nargs='+', type=int, help='compute sa loss in which stages: [0, 1, 2, 3')
		self.parser.add_argument("--mask_pooling_type", default='avg', type=str, help='the manner to downsample predicted masks')

		self.parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add tpavi block in which stages: [0, 1, 2, 3')



		self.parser.add_argument("--weights", type=str, default='', help='path of trained model')
		self.parser.add_argument('--log_dir', default='./train_logs', type=str)
		self.parser.add_argument('--gpu', type=str, default='0,1,2,3,4,5,6,7', help='gpu device number')


		self.parser.add_argument("--tpavi_vv_flag", type=int, default=0, help='visual-visual self-attention')
		self.parser.add_argument("--tpavi_va_flag", type=int, default=1, help='visual-audio cross-attention')

		self.parser.add_argument("--num_workers", default=8, type=int)

		### yb: add --------------->
		self.parser.add_argument("--audio_length", default=1.95, type=float)


		self.parser.add_argument('--Adapter_downsample', type=int, default=2, help="tune top k")
		self.parser.add_argument('--num_conv_group', type=int, default=2, help="group conv")


		self.parser.add_argument('--is_bn', type=int, default=0, help="TF audio adapter")
		self.parser.add_argument('--is_gate', type=int, default=0, help="TF audio adapter")
		self.parser.add_argument('--is_multimodal', type=int, default=1, help="TF audio adapter")
		self.parser.add_argument('--is_before_layernorm', type=int, default=1, help="TF audio adapter")
		self.parser.add_argument('--is_post_layernorm', type=int, default=1, help="TF audio adapter")

		self.parser.add_argument('--is_vit_ln', type=int, default=0, help="TF Vit")

		self.parser.add_argument('--is_fusion_before', type=int, default=0, help="TF Vit")

		self.parser.add_argument('--num_tokens', type=int, default=32, help="num of MBT tokens")

		self.parser.add_argument('--early_stop', type=int, default=5, help='weight and bias setup')

		self.parser.add_argument('--wandb', type=int, default=0 , help='weight and bias setup')
		self.parser.add_argument('--model_name', type=str, default=None, help="for log")
		


		

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()

		str_ids = self.opt.gpu.split(',')
		self.opt.gpu = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu.append(id)

		# # set gpu ids
		# if len(self.opt.gpu_ids) > 0:
		# 	torch.cuda.set_device(self.opt.gpu_ids[0])


		#I should process the opt here, like gpu ids, etc.
		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')


		# save to the disk
		# expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
		# util.mkdirs(expr_dir)
		# file_name = os.path.join(expr_dir, 'opt.txt')
		# with open(file_name, 'wt') as opt_file:
		# 	opt_file.write('------------ Options -------------\n')
		# 	for k, v in sorted(args.items()):
		# 		opt_file.write('%s: %s\n' % (str(k), str(v)))
		# 	opt_file.write('-------------- End ----------------\n')
		return self.opt
