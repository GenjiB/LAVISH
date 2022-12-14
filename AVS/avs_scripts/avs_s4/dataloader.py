import os
from wave import _wave_params
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import torchaudio
import soundfile as sf

import cv2
from PIL import Image
from torchvision import transforms

from config import cfg
from ipdb import set_trace

import warnings
warnings.filterwarnings('ignore')



def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
	img_PIL = Image.open(path).convert(mode)
	if transform:
		img_tensor = transform(img_PIL)
		return img_tensor
	return img_PIL


def load_audio_lm(audio_lm_path):
	with open(audio_lm_path, 'rb') as fr:
		audio_log_mel = pickle.load(fr)
	audio_log_mel = audio_log_mel.detach() # [5, 1, 96, 64]
	return audio_log_mel


class S4Dataset(Dataset):
	"""Dataset for single sound source segmentation"""
	def __init__(self, split='train', args=None):
		super(S4Dataset, self).__init__()
		self.split = split
		self.mask_num = 1 if self.split == 'train' else 5
		df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
		self.df_split = df_all[df_all['split'] == split]
		print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
		self.img_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		])
		self.mask_transform = transforms.Compose([
			transforms.ToTensor(),
		])
		self.opt = args

		# ### ---> yb calculate: AVE dataset for 192
		# self.norm_mean =  -4.984795570373535
		# self.norm_std =  3.7079780101776123
		# ### <----

		### ---> yb calculate: AVE dataset for 192
		self.norm_mean =  -5.210531711578369
		self.norm_std =  3.5918314456939697
		### <----

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
		# if waveform.shape[1] > sr*(self.opt.audio_length+0.1):
		sample_indx = np.linspace(0, waveform.shape[1] -sr*(self.opt.audio_length+0.1), num=5, dtype=int)
		waveform = waveform[:,sample_indx[idx]:sample_indx[idx]+int(sr*self.opt.audio_length)]
		# waveform = waveform.mean(dim=0, keepdim=True)
		## align end ##



		# fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) ## original
		fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=10)


		########### ------> very important: audio normalized
		fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
		### <--------

		# target_length = int(1024 * (self.opt.audio_length/10)) ## for audioset: 10s
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
	

	def __getitem__(self, index):
		df_one_video = self.df_split.iloc[index]
		video_name, category = df_one_video[0], df_one_video[2]
		img_base_path =  os.path.join(cfg.DATA.DIR_IMG, self.split, category, video_name)
		audio_lm_path = os.path.join(cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, category, video_name + '.pkl')
		mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, category, video_name)
		audio_log_mel = load_audio_lm(audio_lm_path)
		# audio_lm_tensor = torch.from_numpy(audio_log_mel)
		imgs, masks = [], []
		for img_id in range(1, 6):
			img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id)), transform=self.img_transform)
			imgs.append(img)
		for mask_id in range(1, self.mask_num + 1):
			mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='1')
			masks.append(mask)
		imgs_tensor = torch.stack(imgs, dim=0)
		masks_tensor = torch.stack(masks, dim=0)

		
		
		### ---> loading all audio frames
		total_audio = []
		for audio_sec in range(5):
			fbank, mix_lambda = self._wav2fbank(os.path.join(cfg.DATA.DIR_AUDIO_WAV, self.split, category, video_name + '.wav'), idx=audio_sec)
			total_audio.append(fbank)
		total_audio = torch.stack(total_audio)
		### <----
		


		if self.split == 'train':
			return imgs_tensor, total_audio, audio_log_mel, masks_tensor
		else:
			return imgs_tensor, total_audio, audio_log_mel, masks_tensor, category, video_name


	def __len__(self):
		return len(self.df_split)




if __name__ == "__main__":
	train_dataset = S4Dataset('train')
	train_dataloader = torch.utils.data.DataLoader(train_dataset,
													 batch_size=2,
													 shuffle=False,
													 num_workers=8,
													 pin_memory=True)

	for n_iter, batch_data in enumerate(train_dataloader):
		imgs, audio, mask = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
		# imgs, audio, mask, category, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
		pdb.set_trace()
	print('n_iter', n_iter)
	pdb.set_trace()
