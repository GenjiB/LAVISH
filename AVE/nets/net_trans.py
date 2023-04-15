import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
from ipdb import set_trace

from torch import Tensor
from typing import Optional, Any
from einops import rearrange, repeat

from timm.models.vision_transformer import Attention
import timm
import loralib as lora
from .my_layers import PHMLinear
from transformers.activations import get_activation

### VGGSound
from nets import Resnet_VGGSound

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


from nets.ast_models import ASTModel
from nets.my_vit import my_vit

def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class VisualAdapter(nn.Module):
	"""Conventional Adapter layer, in which the weights of up and down sampler modules
	are parameters and are optimized."""

	def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None ,use_bn=True, use_gate=True):
		super().__init__()
		self.adapter_kind = adapter_kind
		self.use_bn = use_bn
		self.is_multimodal = opt.is_multimodal
		self.opt = opt

		if use_gate:
			self.gate = nn.Parameter(torch.zeros(1))
		else:
			self.gate = None

		if adapter_kind == "bottleneck" and self.is_multimodal:
			self.down_sample_size = input_dim // reduction_factor
			### -----> attetnion 
			self.my_tokens = nn.Parameter(torch.rand((self.opt.num_tokens, input_dim)))

			self.gate_av = nn.Parameter(torch.zeros(1))

			### <------

			self.activation = nn.ReLU(inplace=True)

			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)

			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group, bias=False)

			if use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)
			
			### -------> yb: add
			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
			### <---------

		elif adapter_kind == "bottleneck":
			self.down_sample_size = input_dim // reduction_factor
			self.activation = nn.ReLU(inplace=True)
			

			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)
			# nn.init.zeros_(self.down_sampler) # yb:for lora

			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group, bias=False)

			if use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)

			### -------> yb: add
			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
			### <---------

		elif adapter_kind == "basic":
			self.activation = nn.ReLU(inplace=True)
			# self.conv = nn.Conv2d(input_dim, output_dim, 1, bias=False)
			self.conv = nn.Linear(input_dim, output_dim, bias=False)

			if use_bn:
				self.bn = nn.BatchNorm1d(output_dim)

		else:
			raise NotImplementedError

	def forward(self, x, vis_token=None):
		if self.adapter_kind == "bottleneck" and self.is_multimodal:
			
			

			### -------> high dim att
			rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))
			att_v2tk = torch.bmm(rep_token, vis_token.squeeze(-1))

			att_v2tk = F.softmax(att_v2tk, dim=-1)
			rep_token_res = torch.bmm(att_v2tk, vis_token.squeeze(-1).permute(0,2,1))

			rep_token = rep_token + rep_token_res
			

			att_tk2x = torch.bmm(x.squeeze(-1).permute(0,2,1), rep_token.permute(0,2,1))

			att_tk2x = F.softmax(att_tk2x, dim=-1)
			x_res = torch.bmm(att_tk2x, rep_token).permute(0,2,1).unsqueeze(-1)


			x = x + self.gate_av*x_res.contiguous()
			### <----------
			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

			z = self.down_sampler(x)

			
			

			


			
			## <----

			if self.use_bn:


				z = self.bn1(z)

			z = self.activation(z)
			output = self.up_sampler(z)
			if self.use_bn:

				output = self.bn2(output)
	
		elif self.adapter_kind == "bottleneck":

			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

			z = self.down_sampler(x)

			if self.use_bn:
				z = self.bn1(z)

			output = self.up_sampler(z)
			if self.use_bn:
				output = self.bn2(output)
			

		elif self.adapter_kind == "basic":
			output = self.conv(x)
			if self.use_bn:
				output = self.bn(rearrange(output, 'N C L -> N L C') )
				output = rearrange(output, 'N L C -> N C L')


		if self.opt.is_post_layernorm:
			output = self.ln_post(output.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

		if self.gate is not None:
			output = self.gate * output



	

		return output




class HyperComplexAdapter(nn.Module):
	"""Hypercomplex Adapter layer, in which the weights of up and down sampler modules
	are parameters are 1/n times of the conventional adapter layers, where n is
	hypercomplex division number."""

	def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None ,use_bn=True, use_gate=True):
		super().__init__()

		import json

		config = json.load(open('/data/yanbo/ada_av/nets/compacter.json'))
		self.input_dim = input_dim
		self.down_sample_size = self.input_dim // reduction_factor

		# self.activation = Activations(self.config['non_linearity'].lower())
		self.activation = get_activation('gelu_new')

		self.down_sampler = PHMLinear(in_features=self.input_dim,
									  out_features=self.down_sample_size,
									  bias=True,
									  c_init=config['phm_c_init'],
									  phm_dim=config['hypercomplex_division'],
									  learn_phm=config['learn_phm'],
									  w_init=config['hypercomplex_nonlinearity'],
									  shared_phm_rule=config['shared_phm_rule'],
									  factorized_phm=config['shared_phm_rule'],
									#   shared_W_phm=config['shared_W_phm'],
									  factorized_phm_rule=config['factorized_phm_rule'],
									#   phm_rank=config['phm_rank'],
									  phm_init_range=config['phm_init_range'],
									#   kronecker_prod=config['kronecker_prod']
									  )
		
		self.up_sampler = PHMLinear(in_features=self.down_sample_size,
									out_features=self.input_dim, 
									bias=True,
									c_init=config['phm_c_init'],
									phm_dim=config['hypercomplex_division'],
									learn_phm=config['learn_phm'],
									w_init=config['hypercomplex_nonlinearity'],
									shared_phm_rule=config['shared_phm_rule'],
									factorized_phm=config['factorized_phm'],
									# shared_W_phm=config['shared_W_phm'],
									factorized_phm_rule=config['factorized_phm_rule'],
									# phm_rank=config['phm_rank'],
									phm_init_range=config['phm_init_range'],
									# kronecker_prod=config['kronecker_prod']
									)

	def forward(self, x):
		z = self.down_sampler(x)
		z = self.activation(z)
		return self.up_sampler(z)

class MMIL_Net(nn.Module):

	def __init__(self, opt):
		super(MMIL_Net, self).__init__()


		self.opt = opt

		# choose which ViTs you want
		if opt.vis_encoder_type == 'swin':
			self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)
			# self.swin = timm.create_model('swinv2_base_window12_192_22k', pretrained=True)
			# self.swin = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
			# self.swin = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
			# self.swin = timm.create_model('swinv2_large_window12to16_192to256_22kft1k', pretrained=True)
			# self.swin = timm.create_model('swinv2_large_window12to24_192to384_22kft1k', pretrained=True)
		elif opt.vis_encoder_type == 'vit':
			self.ViT = my_vit(name='vit_large_patch16_224_in21k')
			# self.ViT = my_vit(name='vit_base_patch16_224_in21k')
			# self.ViT = my_vit(name='vit_tiny_patch16_224_in21k')
			# self.ViT = my_vit(name='vit_small_patch16_224_in21k')



		



		self.mlp_class = nn.Linear(1536*2, 512) # swinv2-Large
		self.mlp_class_2 = nn.Linear(512, 29)

		# self.mlp_class = nn.Linear(1024*2, 512) # swinv2-Base and vit-large
		# self.mlp_class_2 = nn.Linear(512, 29)

		# self.mlp_class = nn.Linear(768*2, 512) # ViT-Base
		# self.mlp_class_2 = nn.Linear(512, 29)

		# self.mlp_class = nn.Linear(384*2, 512) # ViT-small
		# self.mlp_class_2 = nn.Linear(512, 29)

		# self.mlp_class = nn.Linear(192*2, 64) # ViT-tiny
		# self.mlp_class_2 = nn.Linear(64, 29)
		




		
		hidden_list = []
		down_in_dim = []
		down_out_dim = []

		if opt.vis_encoder_type == 'swin':
			## ------------> for swin 
			for idx_layer, my_blk in enumerate(self.swin.layers) :
				if not isinstance(my_blk.downsample, nn.Identity):
					down_in_dim.append(my_blk.downsample.reduction.in_features)
					down_out_dim.append(my_blk.downsample.reduction.out_features)

				for blk in my_blk.blocks:
					hidden_d_size = blk.norm1.normalized_shape[0]
					hidden_list.append(hidden_d_size)

			self.adapter_token_downsampler = nn.ModuleList([
					nn.Linear(down_out_dim[i]//(self.opt.Adapter_downsample*2), down_out_dim[i]//self.opt.Adapter_downsample, bias=False)
					for i in range(len(down_in_dim))])
			self.adapter_token_downsampler.append(nn.Identity())
			## <--------------

		if opt.vis_encoder_type == 'vit':
			### ----------> for vit
			for idx_layer, my_blk in enumerate(self.ViT.v.blocks) :
				hidden_d_size = my_blk.mlp.fc1.in_features
				hidden_list.append(hidden_d_size)
			# ### <---------
		

		if self.opt.is_audio_adapter_p1:
			self.audio_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])



		if self.opt.is_audio_adapter_p2:
			self.audio_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])


		
		

	def forward_swin(self, audio, vis, rand_train_idx=12, stage='eval'):


	   ##### ----------> swin


		vis = rearrange(vis, 'b t c w h -> (b t) c w h')
		f_v = self.swin.patch_embed(vis)
		
		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)

		audio = rearrange(audio, 'b t c w h -> (b t) c w h')
		f_a = self.swin.patch_embed(audio)


		idx_layer = 0
		out_idx_layer = 0
		for _, my_blk in enumerate(self.swin.layers) :


			

			for blk in my_blk.blocks:
				

				f_a_res = self.audio_adapter_blocks_p1[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[idx_layer](f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))

				f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
				f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

				f_a = f_a + blk.drop_path1(blk.norm1(blk._attn(f_a)))
				f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)
				

				

				f_a_res = self.audio_adapter_blocks_p2[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p2[idx_layer]( f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))




				f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
				f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

				f_a = f_a + blk.drop_path2(blk.norm2(blk.mlp(f_a)))
				f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)


				idx_layer = idx_layer +1


			f_v = my_blk.downsample(f_v)
			f_a = my_blk.downsample(f_a)



		f_v = self.swin.norm(f_v)
		f_a = self.swin.norm(f_a)

		f_v = f_v.mean(dim=1, keepdim=True)
		f_a = f_a.mean(dim=1, keepdim=True)


		out_av = torch.cat((f_v, f_a), dim=-1)
		out_av = rearrange(out_av, 'b t p -> (b t) p')



		p_av = self.mlp_class(out_av)
		p_av = self.mlp_class_2(p_av)


		# due to BCEWithLogitsLoss
		p_av = F.softmax(p_av, dim=-1)
		

		return p_av 
		################## <----------- swin

	def forward_vit(self, audio, vis, rand_train_idx=12, stage='eval'):

		b,t,c,w,h = vis.shape

		audio = audio[0]
		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)
		f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True)
		f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True)



		

		bs, dim, wa, ha = patch_info_audio
		bs, dim, wv, wv = patch_info_vis

		layer_count = 0


		for idx_layer, blk in enumerate(self.ViT.v.blocks) :
			# print('yb++++++++++: ', idx_layer)
			if idx_layer >= self.opt.start_tune_layers: 
				

				f_a_res = self.audio_adapter_blocks_p1[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[idx_layer](f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))

				
				f_v = f_v + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(f_v))))
				f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

				f_a = f_a + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(f_a))))
				f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)
				

				f_a_res = self.audio_adapter_blocks_p2[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p2[idx_layer](f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))


	
				f_v = f_v + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_v))))
				f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

				f_a = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_a))))
				f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)



			layer_count += 1



		f_v = self.ViT.v.norm(f_v)
		f_a = self.ViT.v.norm(f_a)

		


		v_cls = f_v[:,0:1].clone()
		a_cls = f_a[:,0:1].clone()



		out_av = torch.cat((a_cls, v_cls), dim=-1)


		out_av = rearrange(out_av, 'b t p -> (b t) p')


		p_av = self.mlp_class(out_av)

		# f_a = rearrange(audio, 'b t dim -> (b t) dim')
		p_av = self.mlp_class_2(p_av)


		# due to BCEWithLogitsLoss
		p_av = F.softmax(p_av, dim=-1)
		

		return p_av 

	def forward(self, audio, vis, rand_train_idx=12, stage='eval'):
		if self.opt.vis_encoder_type == 'swin':
			return self.forward_swin(audio, vis, rand_train_idx=12, stage='eval')
		if self.opt.vis_encoder_type == 'vit':
			return self.forward_vit(audio, vis, rand_train_idx=12, stage='eval')
