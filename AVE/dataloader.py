import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from ipdb import set_trace
import pickle as pkl
import h5py

import soundfile as sf
import torchaudio
import torchvision
import glob


### VGGSound
from scipy import signal
import soundfile as sf
###

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import warnings
warnings.filterwarnings('ignore')



class AVE_dataset(Dataset):

	def __init__(self, opt, mode='train'):

		self.opt = opt
		with h5py.File('./data/labels.h5', 'r') as hf:
			self.labels = hf['avadataset'][:]

		if mode == 'train':
			with h5py.File('data/train_order.h5', 'r') as hf:
				order = hf['order'][:]
		elif mode == 'test':
			with h5py.File('data/test_order.h5', 'r') as hf:
				order = hf['order'][:]

		# with h5py.File('/data/yanbo/ada_av/feats/audio_feature.h5', 'r') as hf:
		# 	self.audio_features = hf['avadataset'][:]

		self.lis = order

		self.raw_gt = pd.read_csv("data/Annotations.txt", sep="&", header=None)

		# with h5py.File('/playpen-iop/yblin/AVE_Dataset/features/audio_feature.h5', 'r') as hf:
		# 	self.audio_features = hf['avadataset'][:]

		# with h5py.File('/playpen-iop/yblin/AVE_Dataset/features/visual_feature.h5', 'r') as hf:
		# 	self.vis_features = hf['avadataset'][:]

		### ---> for audio in AST
		# self.norm_mean = -5.4450
		# self.norm_std = 2.9610
		### <----

		### ---> for audio in AST
		# norm_stats = {'audioset':[-4.2677393, 4.5689974], 
		# 'esc50':[-6.6268077, 5.358466], 
		# 'speechcommands':[-6.845978, 5.5654526]}
		# check ast/src/get_norm_stats.py
		# self.norm_mean = -4.2677393
		# self.norm_std = 4.5689974
		### <----


		### ---> yb calculate: AVE dataset
		if self.opt.vis_encoder_type == 'vit':
			self.norm_mean = -4.1426
			self.norm_std = 3.2001
		### <----
		
		elif self.opt.vis_encoder_type == 'swin':
			## ---> yb calculate: AVE dataset for 192
			self.norm_mean =  -4.984795570373535
			self.norm_std =  3.7079780101776123
			## <----
			

		if self.opt.vis_encoder_type == 'vit':
			self.my_normalize = Compose([
				# Resize([384,384], interpolation=Image.BICUBIC),
				Resize([224,224], interpolation=Image.BICUBIC),
				# Resize([192,192], interpolation=Image.BICUBIC),
				# CenterCrop(224),
				Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
			])
		elif self.opt.vis_encoder_type == 'swin':
			self.my_normalize = Compose([
				Resize([192,192], interpolation=Image.BICUBIC),
				Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
			])
	def getVggoud_proc(self, filename, idx=None):

		audio_length = 1
		samples, samplerate = sf.read(filename)

		if samples.shape[0] > 16000*(audio_length+0.1):
			sample_indx = np.linspace(0, samples.shape[0] -16000*(self.opt.audio_length+0.1), num=10, dtype=int)
			samples = samples[sample_indx[idx]:sample_indx[idx]+int(16000*self.opt.audio_length)]

		else:
			# repeat in case audio is too short
			samples = np.tile(samples,int(self.opt.audio_length))[:int(16000*self.opt.audio_length)]

		samples[samples > 1.] = 1.
		samples[samples < -1.] = -1.

		frequencies, times, spectrogram = signal.spectrogram(samples, samplerate, nperseg=512,noverlap=353)
		spectrogram = np.log(spectrogram+ 1e-7)

		mean = np.mean(spectrogram)
		std = np.std(spectrogram)
		spectrogram = np.divide(spectrogram-mean,std+1e-9)

		
		return torch.tensor(spectrogram).unsqueeze(0).float()

	def _wav2fbank(self, filename, filename2=None, idx=None):
		# mixup
		if filename2 == None:
			waveform, sr = torchaudio.load(filename)
			waveform = waveform - waveform.mean()
		# mixup
		else:
			waveform1, sr = torchaudio.load(filename)
			waveform2, _ = torchaudio.load(filename2)

			waveform1 = waveform1 - waveform1.mean()
			waveform2 = waveform2 - waveform2.mean()

			if waveform1.shape[1] != waveform2.shape[1]:
				if waveform1.shape[1] > waveform2.shape[1]:
					# padding
					temp_wav = torch.zeros(1, waveform1.shape[1])
					temp_wav[0, 0:waveform2.shape[1]] = waveform2
					waveform2 = temp_wav
				else:
					# cutting
					waveform2 = waveform2[0, 0:waveform1.shape[1]]

			# sample lambda from uniform distribution
			#mix_lambda = random.random()
			# sample lambda from beta distribtion
			mix_lambda = np.random.beta(10, 10)

			mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
			waveform = mix_waveform - mix_waveform.mean()
		

		## yb: align ##
		if waveform.shape[1] > 16000*(self.opt.audio_length+0.1):
			sample_indx = np.linspace(0, waveform.shape[1] -16000*(self.opt.audio_length+0.1), num=10, dtype=int)
			waveform = waveform[:,sample_indx[idx]:sample_indx[idx]+int(16000*self.opt.audio_length)]
		## align end ##


		if self.opt.vis_encoder_type == 'vit':
			fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) ## original
			# fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=512, dither=0.0, frame_shift=1)
		elif self.opt.vis_encoder_type == 'swin':
			fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=5.2)

		########### ------> very important: audio normalized
		fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
		### <--------
		if self.opt.vis_encoder_type == 'vit':
			target_length = int(1024 * (1/10)) ## for audioset: 10s
		elif self.opt.vis_encoder_type == 'swin':
			target_length = 192 ## yb: overwrite for swin

		# target_length = 512 ## 5s
		# target_length = 256 ## 2.5s
		n_frames = fbank.shape[0]

		p = target_length - n_frames

		# cut and pad
		if p > 0:
			m = torch.nn.ZeroPad2d((0, 0, 0, p))
			fbank = m(fbank)
		elif p < 0:
			fbank = fbank[0:target_length, :]

		if filename2 == None:
			return fbank, 0
		else:
			return fbank, mix_lambda
	def __len__(self):
		return len(self.lis)

	def __getitem__(self, idx):

		real_idx = self.lis[idx]


		
		file_name = self.raw_gt.iloc[real_idx][1]



		### ---> loading all audio frames
		total_audio = []
		for audio_sec in range(10):
			fbank, mix_lambda = self._wav2fbank(self.opt.audio_folder+'/'+file_name+ '.wav', idx=audio_sec)
			total_audio.append(fbank)
		total_audio = torch.stack(total_audio)
		### <----



		### ---> video frame process 
		total_num_frames = len(glob.glob(self.opt.video_folder+'/'+file_name+'/*.jpg'))
		sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
		total_img = []
		for vis_idx in range(10):
			tmp_idx = sample_indx[vis_idx]
			tmp_img = torchvision.io.read_image(self.opt.video_folder+'/'+file_name+'/'+ str("{:04d}".format(tmp_idx))+ '.jpg')/255
			tmp_img = self.my_normalize(tmp_img)
			total_img.append(tmp_img)
		total_img = torch.stack(total_img)
		### <---


		
		

		
		


		
		return {'audio_spec': total_audio, 
				'GT':self.labels[real_idx], 
				# 'audio_vgg': self.audio_features[real_idx],
				'image':total_img
		}

class ToTensor(object):

	def __call__(self, sample):
		if len(sample) == 2:
			audio = sample['audio']
			label = sample['label']
			return {'audio': torch.from_numpy(audio), 'label': torch.from_numpy(label)}
		else:
			audio = sample['audio']
			video_s = sample['video_s']
			video_st = sample['video_st']
			label = sample['label']
			return {'audio': torch.from_numpy(audio), 'video_s': torch.from_numpy(video_s),
					'video_st': torch.from_numpy(video_st),
					'label': torch.from_numpy(label)}