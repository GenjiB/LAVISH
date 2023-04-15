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

	def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None ,use_bn=True, use_gate=True, num_tk=87):
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
			# self.cm1_att = nn.MultiheadAttention(embed_dim=self.down_sample_size, num_heads=1)

			# self.cm2_att = nn.MultiheadAttention(embed_dim=self.down_sample_size, num_heads=1)




			# self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, self.down_sample_size)))
			# self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, input_dim)))


			self.my_tokens = nn.Parameter(torch.rand((num_tk, input_dim)))

			# self.ln_z = nn.LayerNorm(self.down_sample_size)
			# self.ln_tk = nn.LayerNorm(self.down_sample_size)

			# self.mapping = nn.Conv2d(input_dim, input_dim, 1, groups=self.opt.num_conv_group, bias=False)
			

			self.gate_tk = nn.Parameter(torch.ones(1))


			self.gate_av = nn.Parameter(torch.zeros(1))
	

			
			

			### <------

			self.activation = nn.ReLU(inplace=True)
			# self.down_sampler = nn.Linear(input_dim, self.down_sample_size, bias=False)
			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)
			
			# self.down_sampler_vis = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)

			# self.up_sampler = nn.Linear(self.down_sample_size, output_dim, bias=False)
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
			
			# self.down_sampler = nn.Linear(input_dim, self.down_sample_size, bias=False)
			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)
			nn.init.zeros_(self.down_sampler) # yb:for lora

			# self.up_sampler = nn.Linear(self.down_sample_size, output_dim, bias=False)
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
				# self.bn = nn.BatchNorm2d(output_dim)
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



			#### --------> no MBT cross-modal attention
			# att_vis2x = torch.bmm(x.squeeze(-1).permute(0,2,1), vis_token.squeeze(-1))

			# att_vis2x = F.softmax(att_vis2x, dim=-1)
			# x_res = torch.bmm(att_vis2x, vis_token.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)
			### <------
			x = x + self.gate_av*x_res.contiguous()

			### <----------
			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

			z = self.down_sampler(x)
			
			### -------> Multimodal attention
			# vis_down = self.down_sampler_vis(vis_token)
			# rep_token = repeat(self.my_tokens, 't d -> b t d', b=z.size(0))
			# # cat_toekens = torch.cat((rearrange(rep_token, 'b t d -> t b d'), z.squeeze(-1).permute(2,0,1)),dim=0)


			# ### use vis_down to make rep_toekns' shape -> rep_tokens contain vis features
			# attn_output, _ = self.cm1_att(self.ln_tk(rearrange(rep_token, 'b t d -> t b d')), self.ln_tk(vis_down.squeeze(-1).permute(2,0,1)), self.ln_tk(vis_down.squeeze(-1).permute(2,0,1)))
			# rep_token = attn_output.permute(1,0,2) + rep_token

			# ### use  rep_tokens to make z (audio) shape -> transfer summairzed vis features to audio
			# attn_output_z, _ = self.cm2_att( self.ln_z(z.squeeze(-1).permute(2,0,1)), self.ln_z(rearrange(rep_token, 'b t d -> t b d')), self.ln_z(rearrange(rep_token, 'b t d -> t b d')))
			

			# z = attn_output_z.permute(1,2,0).unsqueeze(-1) + z

			#### <------
			
			

			


			
			## <----

			if self.use_bn:
				# z = self.bn1(rearrange(z, 'N C L -> N L C') )
				# z = rearrange(z, 'N L C -> N C L')

				z = self.bn1(z)

			z = self.activation(z)
			output = self.up_sampler(z)
			if self.use_bn:
				# output = self.bn2(rearrange(output, 'N C L -> N L C') ) 
				# output = rearrange(output, 'N L C -> N C L')
				output = self.bn2(output)
	
		elif self.adapter_kind == "bottleneck":

			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

			z = self.down_sampler(x)

			if self.use_bn:
				z = self.bn1(z)
			# z = self.activation(z)
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

		# self.AST = ASTModel(label_dim=512, fstride=10, tstride=10, input_fdim=128,
		# 						  input_tdim=102, imagenet_pretrain=True,
		# 						  audioset_pretrain=True, model_size='base384') # input_tdim=1024 for 10s 



		self.opt = opt

		# default pretrained=True
		# self.ViT = my_vit()
		
		
		# self.ViT = my_vit(name='vit_tiny_patch16_224_in21k')
		# self.ViT = my_vit(name='vit_base_patch16_224_in21k')
		# self.ViT = my_vit(name='vit_base_patch16_384')
		# vit_tiny_patch16_384
		# vit_base_patch16_384
		# self.ViT = my_vit(name='vit_large_patch32_224_in21k')
		# self.ViT = my_vit(name='deit_base_distilled_patch16_384')
		# self.ViT = my_vit(name='vit_large_patch32_224_in21k')



		# self.ViT = my_vit(name='vit_large_patch16_384')
		# self.ViT = my_vit(name='vit_huge_patch14_224_in21k')



		


		

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



		
		# self.mlp_class = nn.Linear(hidden_d_size*2, 512)
		# self.mlp_class_2 = nn.Linear(512, 29)


		self.mlp_class = nn.Linear(1536*2, 512) # swinv2-Large
		self.mlp_class_2 = nn.Linear(512, 29)
		# self.mlp_map_vggish = nn.Linear(128, 1536) 

		# self.mlp_class = nn.Linear(1024*2, 512) # swinv2-Base and vit-large
		# self.mlp_class_2 = nn.Linear(512, 29)


		# self.mlp_class = nn.Linear(768*2, 512) # ViT-Base
		# self.mlp_class_2 = nn.Linear(512, 29)

		# self.mlp_class = nn.Linear(384*2, 512) # ViT-small
		# self.mlp_class_2 = nn.Linear(512, 29)
		# self.mlp_class = nn.Linear(192*2, 64) # ViT-tiny
		# self.mlp_class_2 = nn.Linear(64, 29)
		

		# self.mlp_class_audio_map = nn.Linear(128, 768)


		
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
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], 
				adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,
				reduction_factor=self.opt.Adapter_downsample, 
				opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate,
				num_tk=opt.num_tokens
				)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], 
				output_dim=hidden_list[i], adapter_kind="bottleneck", 
				dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, 
				opt=opt, use_bn=self.opt.is_bn, use_gate=True,
				num_tk=opt.num_tokens)
				for i in range(len(hidden_list))])

			#### ------> LoRa
			# self.audio_adapter_blocks_p1 = nn.ModuleList([lora.Linear(in_features=hidden_list[i], out_features=hidden_list[i], r=self.opt.Adapter_downsample)
			# for i in range(len(hidden_list))])

			# self.vis_adapter_blocks_p1 = nn.ModuleList([lora.Linear(in_features=hidden_list[i], out_features=hidden_list[i], r=self.opt.Adapter_downsample)
			# for i in range(len(hidden_list))])
			# ##### <------

			#### ------> compacter
			# self.audio_adapter_blocks_p1 = nn.ModuleList([
			# 	HyperComplexAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
			# for i in range(len(hidden_list))])

			# self.vis_adapter_blocks_p1 = nn.ModuleList([
			# 	HyperComplexAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
			# for i in range(len(hidden_list))])


			# self.audio_adapter_blocks_p1_gate = nn.Parameter(torch.zeros(1))
			# self.vis_adapter_blocks_p1_gate = nn.Parameter(torch.zeros(1))
			# ##### <------

		if self.opt.is_audio_adapter_p2:
			self.audio_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", 
				dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, 
				opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate,
				num_tk=opt.num_tokens)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", 
				dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, 
				opt=opt, use_bn=self.opt.is_bn, use_gate=True,
				num_tk=opt.num_tokens)
				for i in range(len(hidden_list))])

			#### ------> LoRa
			# self.audio_adapter_blocks_p2 = nn.ModuleList([lora.Linear(in_features=hidden_list[i], out_features=hidden_list[i], r=self.opt.Adapter_downsample)
			# for i in range(len(hidden_list))])

			# self.vis_adapter_blocks_p2 = nn.ModuleList([lora.Linear(in_features=hidden_list[i], out_features=hidden_list[i], r=self.opt.Adapter_downsample)
			# for i in range(len(hidden_list))])
			# ##### <------

			#### ------> compacter
			# self.audio_adapter_blocks_p2 = nn.ModuleList([
			# 	HyperComplexAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
			# for i in range(len(hidden_list))])

			# self.vis_adapter_blocks_p2 = nn.ModuleList([
			# 	HyperComplexAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
			# for i in range(len(hidden_list))])


			# self.audio_adapter_blocks_p2_gate = nn.Parameter(torch.zeros(1))
			# self.vis_adapter_blocks_p2_gate = nn.Parameter(torch.zeros(1))
			# ##### <------


		
		

			

	def forward_joint_vis_first(self, audio, vis):
		b,t,c,w,h = vis.shape


		f_v = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'))

		f_a = self.AST(rearrange(audio, 'b t len dim -> (b t) len dim'), additional_patch=f_v)
		additional_idx = f_v.shape[1]

		a_cls = ((f_a[:,0:1] + f_a[:,1:2])/2)
		v_cls = (f_a[:,-additional_idx:-additional_idx+1] + f_a[:,-additional_idx+1:-additional_idx+2])/2

		
		

		# out_a,_ = self.lstm_audio(rearrange(a_cls, '(b t) 1 d -> b t d', b=b,t=t))

		out_av = torch.cat((a_cls, v_cls), dim=-1)




		out_av = rearrange(out_av, 'b t p -> (b t) p')
		p_av = self.mlp_class(out_av)

		# f_a = rearrange(audio, 'b t dim -> (b t) dim')
		p_av = self.mlp_class_2(p_av)

		p_av = F.softmax(p_av, dim=-1)
		
		return p_av

	def forward323(self, audio, vis, rand_train_idx=12, stage='eval'):
		b,t,c,w,h = vis.shape

		f_v = self.ViT.forward_features(rearrange(vis, 'b t c w h -> (b t) c w h'))

		f_a = self.AST(rearrange(audio, 'b t len dim -> (b t) len dim'))		
		

		# out_a,_ = self.lstm_audio(rearrange(a_cls, '(b t) 1 d -> b t d', b=b,t=t))


		out_av = torch.cat((a_cls, v_cls), dim=-1)




		out_av = rearrange(out_av, 'b t p -> (b t) p')
		p_av = self.mlp_class(out_av)

		# f_a = rearrange(audio, 'b t dim -> (b t) dim')
		p_av = self.mlp_class_2(p_av)

		p_av = F.softmax(p_av, dim=-1)
		
		return p_av

	def forward44(self, audio, vis, rand_train_idx=12, stage='eval'):
		# resnet baseline
		b,t,c,w,h = vis.shape


		######### ---------> Timm model
		f_v = self.ViT.forward_features(rearrange(vis, 'b t c w h -> (b t) c w h'))

		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3) #AST pre-proc
		# audio = repeat(audio, 'b t c len dim -> b t (repeat c) len dim', repeat=3) #VGGSound pre-proc
		f_a = self.ViT.forward_features(rearrange(audio, 'b t c len dim -> (b t) c len dim'))

		f_v = F.adaptive_avg_pool2d(f_v,1).squeeze(-1).squeeze(-1)
		f_a = F.adaptive_avg_pool2d(f_a,1).squeeze(-1).squeeze(-1)
		####### <-------


		##### ---------> VGGsound 
		# vis = vis.mean(dim=2, keepdim=True)
		# f_v = self.vggsound(rearrange(vis, 'b t c len dim -> (b t) c len dim'))# b x 512

		# f_a = self.vggsound(rearrange(audio, 'b t c len dim -> (b t) c len dim')) # b x 512

		###### <--------


		out_av = torch.cat((f_a, f_a), dim=-1)



		# set_trace()
		# out_av = rearrange(out_av, 'b t p -> (b t) p')
		p_av = self.mlp_class(out_av)

		# f_a = rearrange(audio, 'b t dim -> (b t) dim')
		p_av = self.mlp_class_2(p_av)

		p_av = F.softmax(p_av, dim=-1)
		
		return p_av
	def forward_joint_audio_first(self, audio, vis):



		b,t,c,w,h = vis.shape


		with torch.no_grad():
			teacher_a_cls = self.AST(rearrange(audio, 'b t len dim -> (b t) len dim'))
			teacher_v_cls = self.ViT.forward_features(rearrange(vis, 'b t c w h -> (b t) c w h'))

		f_a = self.AST.forward_patch(rearrange(audio, 'b t len dim -> (b t) len dim'))

		f_v = self.ViT.forward_features(rearrange(vis, 'b t c w h -> (b t) c w h'), additional_patch=f_a)

		# f_v = self.ViT.forward_features(rearrange(vis, 'b t c w h -> (b t) c w h'))
		additional_idx = f_a.shape[1]

		
		v_cls = ((f_v[:,0:1] + f_v[:,1:2])/2)
		a_cls = (f_v[:,-additional_idx:-additional_idx+1] + f_v[:,-additional_idx+1:-additional_idx+2])/2

		
		

		# out_a,_ = self.lstm_audio(rearrange(a_cls, '(b t) 1 d -> b t d', b=b,t=t))

		out_av = torch.cat((a_cls, v_cls), dim=-1)




		out_av = rearrange(out_av, 'b t p -> (b t) p')
		p_av = self.mlp_class(out_av)

		# f_a = rearrange(audio, 'b t dim -> (b t) dim')
		p_av = self.mlp_class_2(p_av)

		p_av = F.softmax(p_av, dim=-1)
		

		return p_av, (a_cls, teacher_a_cls.unsqueeze(1), v_cls, teacher_v_cls)

	def forward_swin(self, audio, vis, rand_train_idx=12, stage='eval'):



		# vggish = audio[1]
		# vggish = rearrange(vggish, 'b t d-> (b t) 1 d')
		
		audio = audio[0]
	   ##### ----------> swin


		vis = rearrange(vis, 'b t c w h -> (b t) c w h')
		f_v = self.swin.patch_embed(vis)
		
		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)


		### ---------> yb: either chose one
		# audio = F.pad(audio, (48, 48, 61, 61) , "constant", 0)
		# audio = F.interpolate(rearrange(audio, 'b t c w h -> b c t w h'), mode='trilinear',size=[10,192,192])
		####### <-------
		##### audio = rearrange(audio, 'b c t w h -> b t c w h ')
		audio = rearrange(audio, 'b t c w h -> (b t) c w h')
		f_a = self.swin.patch_embed(audio)


		# v_cls = f_v.mean(dim=1, keepdim=True)
		# a_cls = f_a.mean(dim=1, keepdim=True)

		# f_v = torch.cat((v_cls, f_v), dim=1)
		# f_a = torch.cat((a_cls, f_a), dim=1)
		
		idx_layer = 0
		out_idx_layer = 0
		for _, my_blk in enumerate(self.swin.layers) :


			# f_v = blk.blocks[0].attn(f_v)
			# f_v = blk.blocks[0].mlp(f_v)
			# f_v = blk.blocks[1].attn(f_v)
			# f_v = blk.blocks[1].mlp(f_v)


			#######
			# for blk in my_blk.blocks:

			

			for blk in my_blk.blocks:
				
				# self.vis_adapter_blocks_p1[idx_layer].is_multimodal = False
				f_a_res = self.audio_adapter_blocks_p1[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[idx_layer](f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))

				#### ---------> LoRA
				# f_a_res = self.audio_adapter_blocks_p1[idx_layer](f_a)* self.audio_adapter_blocks_p1_gate
				# f_v_res = self.vis_adapter_blocks_p1[idx_layer](f_v) * self.vis_adapter_blocks_p1_gate
				### <--------


				f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
				f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

				f_a = f_a + blk.drop_path1(blk.norm1(blk._attn(f_a)))
				f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)
				

				
				# self.vis_adapter_blocks_p2[idx_layer].is_multimodal = False
				f_a_res = self.audio_adapter_blocks_p2[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p2[idx_layer]( f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))

				#### ---------> LoRA
				# f_a_res = self.audio_adapter_blocks_p2[idx_layer](f_a) * self.audio_adapter_blocks_p2_gate
				# f_v_res = self.vis_adapter_blocks_p2[idx_layer](f_v) * self.vis_adapter_blocks_p2_gate
				### <--------


				f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
				f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

				f_a = f_a + blk.drop_path2(blk.norm2(blk.mlp(f_a)))
				f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)


				idx_layer = idx_layer +1


			# my_tokens_a = self.adapter_token_downsampler[out_idx_layer](my_tokens_a)
			# my_tokens_v = self.adapter_token_downsampler[out_idx_layer](my_tokens_v)
			# out_idx_layer = out_idx_layer + 1
			# if self.audio_adapter_blocks_p2[0].my_tokens.sum() != 0:
			# print(self.audio_adapter_blocks_p2[0].gate_tk)
			f_v = my_blk.downsample(f_v)
			f_a = my_blk.downsample(f_a)



		f_v = self.swin.norm(f_v)
		f_a = self.swin.norm(f_a)

		f_v = f_v.mean(dim=1, keepdim=True)
		f_a = f_a.mean(dim=1, keepdim=True)

		
		# audio = F.pad(audio, (48, 48, 61, 61) , "constant", 0)

		# f_a = self.swin.forward_features(rearrange(audio, 'b t c w h -> (b t) c w h'))
		


		#### --------> yb: only for vggish features
		# f_a = rearrange(audio, 'b t d -> (b t) 1 d')
		# f_a = self.mlp_class_audio_map(f_a)
		# ######## <-------
		# vggish = self.mlp_map_vggish(vggish)
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
		
		f_a_ast = self.AST(rearrange(audio, 'b t len dim -> (b t) len dim'))

		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)
		f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True)
		f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True)



		

		bs, dim, wa, ha = patch_info_audio
		bs, dim, wv, wv = patch_info_vis

		# 

		# additional_idx = f_a.shape[1]
		layer_count = 0


		#### --------> for joint exp
		# all_tokens = torch.cat((f_a, f_v),dim=1)
		### <-------
		for idx_layer, blk in enumerate(self.ViT.v.blocks) :
			# print('yb++++++++++: ', idx_layer)
			if idx_layer >= self.opt.start_tune_layers: 
				
				# self.audio_adapter_blocks_p1[idx_layer].is_multimodal = False
				f_a_res = self.audio_adapter_blocks_p1[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[idx_layer](f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))

				
				f_v = f_v + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(f_v))))
				f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

				f_a = f_a + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(f_a))))
				f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)
				

				# self.audio_adapter_blocks_p2[idx_layer].is_multimodal = False
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

	
		#### --------> yb: only for vggish features
		# a_cls = rearrange(audio, 'b t d -> (b t) 1 d')
		# a_cls = self.mlp_class_audio_map(a_cls)
		# ######## <-------


		
		# out_av = torch.cat((f_a_ast.unsqueeze(1), v_cls), dim=-1)
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
