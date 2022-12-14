import torch
import torch.nn as nn
import torchvision.models as models
from model.pvt import pvt_v2_b5
from model.TPAVI import TPAVIModule
from ipdb import set_trace
import timm
import torch.nn.functional as F
from einops import rearrange, repeat


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



			self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, input_dim)))
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
			x = x + self.gate_av*x_res.contiguous()

			### <----------
			
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

			z = self.activation(z)
			output = self.up_sampler(z)
			if self.use_bn:
				output = self.bn2(output)
			

		elif self.adapter_kind == "basic":
			output = self.conv(x)
			if self.use_bn:
				output = self.bn(rearrange(output, 'N C L -> N L C') )
				output = rearrange(output, 'N L C -> N C L')

		if self.gate is not None:
			output = self.gate * output


		if self.opt.is_post_layernorm:
			output = self.ln_post(output.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)
	

		return output

class Classifier_Module(nn.Module):
	def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
		super(Classifier_Module, self).__init__()
		self.conv2d_list = nn.ModuleList()
		for dilation, padding in zip(dilation_series, padding_series):
			self.conv2d_list.append(nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
		for m in self.conv2d_list:
			m.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.conv2d_list[0](x)
		for i in range(len(self.conv2d_list)-1):
			out += self.conv2d_list[i+1](x)
		return out


class BasicConv2d(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
		super(BasicConv2d, self).__init__()
		self.conv_bn = nn.Sequential(
			nn.Conv2d(in_planes, out_planes,
					  kernel_size=kernel_size, stride=stride,
					  padding=padding, dilation=dilation, bias=False),
			nn.BatchNorm2d(out_planes)
		)

	def forward(self, x):
		x = self.conv_bn(x)
		return x


class ResidualConvUnit(nn.Module):
	"""Residual convolution module.
	"""

	def __init__(self, features):
		"""Init.
		Args:
			features (int): number of features
		"""
		super().__init__()

		self.conv1 = nn.Conv2d(
			features, features, kernel_size=3, stride=1, padding=1, bias=True
		)
		self.conv2 = nn.Conv2d(
			features, features, kernel_size=3, stride=1, padding=1, bias=True
		)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		"""Forward pass.
		Args:
			x (tensor): input
		Returns:
			tensor: output
		"""
		out = self.relu(x)
		out = self.conv1(out)
		out = self.relu(out)
		out = self.conv2(out)

		return out + x

class FeatureFusionBlock(nn.Module):
	"""Feature fusion block.
	"""

	def __init__(self, features):
		"""Init.
		Args:
			features (int): number of features
		"""
		super(FeatureFusionBlock, self).__init__()

		self.resConfUnit1 = ResidualConvUnit(features)
		self.resConfUnit2 = ResidualConvUnit(features)

	def forward(self, *xs):
		"""Forward pass.
		Returns:
			tensor: output
		"""
		output = xs[0]

		if len(xs) == 2:
			output += self.resConfUnit1(xs[1])

		output = self.resConfUnit2(output)

		output = nn.functional.interpolate(
			output, scale_factor=2, mode="bilinear", align_corners=True
		)

		return output


class Interpolate(nn.Module):
	"""Interpolation module.
	"""

	def __init__(self, scale_factor, mode, align_corners=False):
		"""Init.
		Args:
			scale_factor (float): scaling
			mode (str): interpolation mode
		"""
		super(Interpolate, self).__init__()

		self.interp = nn.functional.interpolate
		self.scale_factor = scale_factor
		self.mode = mode
		self.align_corners = align_corners

	def forward(self, x):
		"""Forward pass.
		Args:
			x (tensor): input
		Returns:
			tensor: interpolated data
		"""

		x = self.interp(
			x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
		)

		return x

class Pred_endecoder(nn.Module):
	# pvt-v2 based encoder decoder
	def __init__(self, channel=256,opt=None, config=None, vis_dim=[64, 128, 320, 512], tpavi_stages=[], tpavi_vv_flag=False, tpavi_va_flag=True):
		super(Pred_endecoder, self).__init__()
		self.cfg = config
		self.tpavi_stages = tpavi_stages
		self.tpavi_vv_flag = tpavi_vv_flag
		self.tpavi_va_flag = tpavi_va_flag
		self.vis_dim = vis_dim

		self.opt = opt
		
		self.relu = nn.ReLU(inplace=True)

		self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
		self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
		self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
		self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

		self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[3])
		self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[2])
		self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[1])
		self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[0])

		self.path4 = FeatureFusionBlock(channel)
		self.path3 = FeatureFusionBlock(channel)
		self.path2 = FeatureFusionBlock(channel)
		self.path1 = FeatureFusionBlock(channel)



		self.x1_linear = nn.Linear(192,64)
		self.x2_linear = nn.Linear(384,128)
		self.x3_linear = nn.Linear(768,320)
		self.x4_linear = nn.Linear(1536,512)

		self.audio_linear = nn.Linear(1536,128)

		self.encoder_backbone = pvt_v2_b5()

		self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)
		# self.swin = timm.create_model('swinv2_large_window12to16_192to256_22kft1k', pretrained=True)


		### ------------> for swin 
		hidden_list = []
		down_in_dim = []
		down_out_dim = []
		for idx_layer, my_blk in enumerate(self.swin.layers) :
			if not isinstance(my_blk.downsample, nn.Identity):
				down_in_dim.append(my_blk.downsample.reduction.in_features)
				down_out_dim.append(my_blk.downsample.reduction.out_features)

			for blk in my_blk.blocks:
				hidden_d_size = blk.norm1.normalized_shape[0]
				hidden_list.append(hidden_d_size)
		### <--------------


		self.audio_adapter_blocks_p1 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
			for i in range(len(hidden_list))])

		self.vis_adapter_blocks_p1 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
			for i in range(len(hidden_list))])

		self.audio_adapter_blocks_p2 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
			for i in range(len(hidden_list))])

		self.vis_adapter_blocks_p2 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
			for i in range(len(hidden_list))])

		

		for i in self.tpavi_stages:
			setattr(self, f"tpavi_b{i+1}", TPAVIModule(in_channels=channel, mode='dot'))
			print("==> Build TPAVI block...")

		self.output_conv = nn.Sequential(
			nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
			Interpolate(scale_factor=2, mode="bilinear"),
			nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(True),
			nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
		)

		if self.training:
			self.initialize_pvt_weights()


	def pre_reshape_for_tpavi(self, x):
		# x: [B*5, C, H, W]
		_, C, H, W = x.shape
		try:
			x = x.reshape(-1, 5, C, H, W)
		except:
			print("pre_reshape_for_tpavi: ", x.shape)
		x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
		return x

	def post_reshape_for_tpavi(self, x):
		# x: [B, C, T, H, W]
		# return: [B*T, C, H, W]
		_, C, _, H, W = x.shape
		x = x.permute(0, 2, 1, 3, 4) # [B, T, C, H, W]
		x = x.view(-1, C, H, W)
		return x

	def tpavi_vv(self, x, stage):
		# x: visual, [B*T, C=256, H, W]
		tpavi_b = getattr(self, f'tpavi_b{stage+1}')
		x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
		x, _ = tpavi_b(x) # [B, C, T, H, W]
		x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
		return x

	def tpavi_va(self, x, audio, stage):
		# x: visual, [B*T, C=256, H, W]
		# audio: [B*T, 128]
		# ra_flag: return audio feature list or not
		tpavi_b = getattr(self, f'tpavi_b{stage+1}')
		try:
			audio = audio.view(-1, 5, audio.shape[-1]) # [B, T, 128]
		except:
			print("tpavi_va: ", audio.shape)
		x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
		x, a = tpavi_b(x, audio) # [B, C, T, H, W], [B, T, C]
		x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
		return x, a

	def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
		return block(dilation_series, padding_series, NoLabels, input_channel)

	def forward(self, x, audio_feature=None):
		# ---------> yb:add
		B, frame, C, H, W = x.shape
		x = x.view(B*frame, C, H, W)

		# audio_feature = audio_feature.view(-1, 5, audio_feature.shape[-1])



		audio = repeat(audio_feature, 'b t len dim -> b t c len dim', c=3)
		audio = rearrange(audio, 'b t c w h -> (b t) c w h')
		f_a = self.swin.patch_embed(audio)

		x = F.interpolate(x, mode='bicubic',size=[192,192])
		f_v = self.swin.patch_embed(x)
		
		idx_layer = 0
		multi_scale = []
		
		idx_block = 0
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

			if idx_block != 3:
				multi_scale.append(f_v)
			else:
				multi_scale.append(self.swin.norm(f_v))
			f_v = my_blk.downsample(f_v)
			f_a = my_blk.downsample(f_a)
			idx_block += 1

		f_v = self.swin.norm(f_v)
		f_a = self.swin.norm(f_a)

		

		audio_feature = rearrange(f_a.mean(dim=1), '(b t) d -> b t d', t=5)
		audio_feature = self.audio_linear(audio_feature)


		############
		## torch.Size([20, 2304, 192])
		## torch.Size([20, 576, 384])
		## torch.Size([20, 144, 768])
		## torch.Size([20, 36, 1536])


		x = self.swin.norm(x)

		x = x.mean(dim=1, keepdim=True)

		#  <-------



		# x1, x2, x3, x4 = self.encoder_backbone(x)
		# print(x1.shape, x2.shape, x3.shape, x4.shape)
		# shape for pvt-v2-b5
		# BF x  64 x 56 x 56
		# BF x 128 x 28 x 28
		# BF x 320 x 14 x 14
		# BF x 512 x  7 x  7
		x1 = multi_scale[0].view(multi_scale[0].size(0),48,48,-1)
		x2 = multi_scale[1].view(multi_scale[1].size(0),24,24,-1)
		x3 = multi_scale[2].view(multi_scale[2].size(0),12,12,-1)
		x4 = multi_scale[3].view(multi_scale[3].size(0),6,6,-1)


		x1 = multi_scale[0].view(multi_scale[0].size(0),64,64,-1)
		x2 = multi_scale[1].view(multi_scale[1].size(0),32,32,-1)
		x3 = multi_scale[2].view(multi_scale[2].size(0),16,16,-1)
		x4 = multi_scale[3].view(multi_scale[3].size(0),8,8,-1)

		x1 = self.x1_linear(x1)
		x2 = self.x2_linear(x2)
		x3 = self.x3_linear(x3)
		x4 = self.x4_linear(x4)

		x1 = F.interpolate(rearrange(x1, 'BF w h c -> BF c w h'), mode='bicubic',size=[56,56])
		x2 = F.interpolate(rearrange(x2, 'BF w h c -> BF c w h'), mode='bicubic',size=[28,28])
		x3 = F.interpolate(rearrange(x3, 'BF w h c -> BF c w h'), mode='bicubic',size=[14,14])
		x4 = F.interpolate(rearrange(x4, 'BF w h c -> BF c w h'), mode='bicubic',size=[7,7])


		
		conv1_feat = self.conv1(x1)    # BF x 256 x 56 x 56
		conv2_feat = self.conv2(x2)    # BF x 256 x 28 x 28
		conv3_feat = self.conv3(x3)    # BF x 256 x 14 x 14
		conv4_feat = self.conv4(x4)    # BF x 256 x  7 x  7
		# print(conv1_feat.shape, conv2_feat.shape, conv3_feat.shape, conv4_feat.shape)

		feature_map_list = [conv1_feat, conv2_feat, conv3_feat, conv4_feat]
		a_fea_list = [None] * 4

		if len(self.tpavi_stages) > 0:
			if (not self.tpavi_vv_flag) and (not self.tpavi_va_flag):
				raise Exception('tpavi_vv_flag and tpavi_va_flag cannot be False at the same time if len(tpavi_stages)>0, \
					tpavi_vv_flag is for video self-attention while tpavi_va_flag indicates the standard version (audio-visual attention)')
			for i in self.tpavi_stages:
				tpavi_count = 0
				conv_feat = torch.zeros_like(feature_map_list[i]).cuda()
				if self.tpavi_vv_flag:
					conv_feat_vv = self.tpavi_vv(feature_map_list[i], stage=i)
					conv_feat += conv_feat_vv
					tpavi_count += 1
				if self.tpavi_va_flag:
					conv_feat_va, a_fea = self.tpavi_va(feature_map_list[i], audio_feature, stage=i)
					conv_feat += conv_feat_va
					tpavi_count += 1
					a_fea_list[i] = a_fea
				conv_feat /= tpavi_count
				feature_map_list[i] = conv_feat # update features of stage-i which conduct non-local

		conv4_feat = self.path4(feature_map_list[3])            # BF x 256 x 14 x 14
		conv43 = self.path3(conv4_feat, feature_map_list[2])    # BF x 256 x 28 x 28
		conv432 = self.path2(conv43, feature_map_list[1])       # BF x 256 x 56 x 56
		conv4321 = self.path1(conv432, feature_map_list[0])     # BF x 256 x 112 x 112

		pred = self.output_conv(conv4321)   # BF x 1 x 224 x 224
		# print(pred.shape)

		return pred, feature_map_list, a_fea_list


	def initialize_pvt_weights(self,):
		pvt_model_dict = self.encoder_backbone.state_dict()
		pretrained_state_dicts = torch.load(self.cfg.TRAIN.PRETRAINED_PVTV2_PATH)
		# for k, v in pretrained_state_dicts['model'].items():
		#     if k in pvt_model_dict.keys():
		#         print(k, v.requires_grad)
		state_dict = {k : v for k, v in pretrained_state_dicts.items() if k in pvt_model_dict.keys()}
		pvt_model_dict.update(state_dict)
		self.encoder_backbone.load_state_dict(pvt_model_dict)
		print(f'==> Load pvt-v2-b5 parameters pretrained on ImageNet from {self.cfg.TRAIN.PRETRAINED_PVTV2_PATH}')
		# pdb.set_trace()


if __name__ == "__main__":
	imgs = torch.randn(10, 3, 224, 224)
	audio = torch.randn(2, 5, 128)
	# model = Pred_endecoder(channel=256)
	model = Pred_endecoder(channel=256, tpavi_stages=[0,1,2,3], tpavi_va_flag=True,)
	# output = model(imgs)
	output = model(imgs, audio)
	pdb.set_trace()