import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace
import numpy as np

class BaseLoss(nn.Module):
	def __init__(self):
		super(BaseLoss, self).__init__()

	def forward(self, preds, targets, weight=None):
		if isinstance(preds, list):
			N = len(preds)
			if weight is None:
				weight = preds[0].new_ones(1)

			errs = [self._forward(preds[n], targets[n], weight[n])
					for n in range(N)]
			err = torch.mean(torch.stack(errs))

		elif isinstance(preds, torch.Tensor):
			if weight is None:
				weight = preds.new_ones(1)
			err = self._forward(preds, targets, weight)

		return err
	
class SmoothL1Loss(BaseLoss):
	def __init__(self):
		super(SmoothL1Loss, self).__init__()

	def _forward(self, pred, target, weight):
		return F.smooth_l1_loss(pred, target)

class L1Loss(BaseLoss):
	def __init__(self):
		super(L1Loss, self).__init__()

	def _forward(self, pred, target, weight):
		return torch.mean(weight * torch.abs(pred - target))

class L2Loss(BaseLoss):
	def __init__(self):
		super(L2Loss, self).__init__()

	def _forward(self, pred, target, weight):
		return torch.mean(weight * torch.pow(pred - target, 2))

class BCELoss(BaseLoss):
	def __init__(self):
		super(BCELoss, self).__init__()

	def _forward(self, pred, target, weight):
		return F.binary_cross_entropy(pred, target, weight=weight)

class BCEWithLogitsLoss(BaseLoss):
	def __init__(self):
		super(BCEWithLogitsLoss, self).__init__()

	def _forward(self, pred, target, weight):
		return F.binary_cross_entropy_with_logits(pred, target, weight=weight)

class CELoss(BaseLoss):
	def __init__(self):
		super(CELoss, self).__init__()

	def _forward(self, pred, target, weight=None):
		return F.cross_entropy(pred, target)

class YBLoss2(torch.nn.Module):
	"""
	Contrastive loss function.
	Based on:
	"""
	def __init__(self, margin=1.0, τ=0.05):
		super(YBLoss2, self).__init__()
		self.margin = margin
		self.τ = τ
	def forward(self,prob_x1,prob_x2, prob_joint,rand_idx,sample_idx, target,opt,x1,x2, x1_class, x2_class):

		v_prob_pos = torch.zeros(prob_x2.shape[0],1,25).to(prob_x2.device)
		v_prob_neg = torch.zeros(prob_x2.shape[0],1,25).to(prob_x2.device)
		a_prob_pos = torch.zeros(prob_x2.shape[0],1,25).to(prob_x2.device)
		a_prob_neg = torch.zeros(prob_x2.shape[0],1,25).to(prob_x2.device)

		feature_dic1= []
		feature_dic2  = []

		class_dic1= []
		class_dic2= []

		
		loss = []

		x1_prob_total = []
		x2_prob_total = []
		
		for i in range(x1.size(0)):
			# if (target[audio_idx[i]] * target[vis_idx[i]]).sum()==0:

				# if opt.exp:
				#     a_prob_pos[audio_idx[i]] = torch.exp(prob_a[audio_idx[i]])
				#     v_prob_pos[vis_idx[i]] = torch.exp(prob_v[vis_idx[i]])

				#     a_prob_neg[audio_idx[i]] += torch.exp(prob_v[-len(audio_idx)+i])
				#     v_prob_neg[vis_idx[i]] += torch.exp(prob_a[-len(audio_idx)+i])
	  
				# else:
			if opt.aug_type=='vision':
				# a_prob_pos[audio_idx[i]] = prob_a[audio_idx[i]]
				# v_prob_pos[vis_idx[i]] = prob_v[vis_idx[i]]
				loss.append(F.binary_cross_entropy(prob_x2[i], target[sample_idx[i]]))



				# a_prob_neg[audio_idx[i]] += prob_v[-len(audio_idx)+i]
				# v_prob_neg[vis_idx[i]] += output[-len(audio_idx)+i]
			elif opt.aug_type=='audio':
				# a_prob_pos[vis_idx[i]] = prob_a[vis_idx[i]]

				loss.append(F.binary_cross_entropy(prob_x2[i], target[sample_idx[i]]))
				# a_prob_neg[audio_idx[i]] += prob_v[-len(audio_idx)+i]
				# a_prob_neg[vis_idx[i]] += output[-len(audio_idx)+i]
				# loss.append(F.binary_cross_entropy(prob_a[vis_idx[i]], target[vis_idx[i]]))
			elif opt.aug_type=='ada':

				if i == 0:

					feature_dic1 = x1[i].unsqueeze(0)
					feature_dic2 = x2[i].unsqueeze(0)

					
					class_dic1 = x1_class[i].unsqueeze(0)
					class_dic2 = x2_class[i].unsqueeze(0)

				else:
					feature_dic1 = torch.cat((feature_dic1, x1[i].unsqueeze(0)), dim=0)
					feature_dic2 = torch.cat((feature_dic2, x2[i].unsqueeze(0)), dim=0)

					class_dic1 = torch.cat((class_dic1, x1_class[i].unsqueeze(0)), dim=0)
					class_dic2 = torch.cat((class_dic2, x2_class[i].unsqueeze(0)), dim=0)
					

				

			elif opt.aug_type=='mix' or opt.aug_type=='yybag':
				
				
				gg_sample = torch.cat((
						   (prob_x2[i]*target[sample_idx[i]]).max().unsqueeze(0),
						   (prob_x2[i+1*len(rand_idx)]*target[sample_idx[i]]).max().unsqueeze(0),
						   (prob_x2[i+2*len(rand_idx)]*target[sample_idx[i]]).max().unsqueeze(0),
						   (prob_x2[i+3*len(rand_idx)]*target[sample_idx[i]]).max().unsqueeze(0)
						   ))
				gg_rand = torch.cat((
						   (prob_x1[i]*target[rand_idx[i]]).max().unsqueeze(0),
						   (prob_x1[i+1*len(rand_idx)]*target[rand_idx[i]]).max().unsqueeze(0),
						   (prob_x1[i+2*len(rand_idx)]*target[rand_idx[i]]).max().unsqueeze(0),
						   (prob_x1[i+3*len(rand_idx)]*target[rand_idx[i]]).max().unsqueeze(0)
						   ))
		
				# joint_label = target[rand_idx[i]] + target[sample_idx[i]]
				# joint_label[joint_label!=0] = 1
				# gg_joint = torch.cat((
				# 		   (prob_x1[i]*joint_label).max().unsqueeze(0),
				# 		   (prob_x1[i+1*len(rand_idx)]*joint_label).max().unsqueeze(0),
				# 		   (prob_x1[i+2*len(rand_idx)]*joint_label).max().unsqueeze(0),
				# 		   (prob_x1[i+3*len(rand_idx)]*joint_label).max().unsqueeze(0)
				# 		   ))

				# if len(feature_dic1) == 0:

				#     feature_dic1 = x1[i+gg_rand.argmax()*len(rand_idx)].unsqueeze(0)
				#     feature_dic2 = x2[i+gg_sample.argmax()*len(rand_idx)].unsqueeze(0)
				# else:

				#     feature_dic1 = torch.cat((feature_dic1, x1[i+gg_rand.argmax()*len(rand_idx)].unsqueeze(0)), dim=0)
				#     feature_dic2 = torch.cat((feature_dic2, x2[i+gg_sample.argmax()*len(rand_idx)].unsqueeze(0)), dim=0)
					
				# x1_prob_total.append(prob_x1[i+gg_rand.argmax()*len(rand_idx)])
				# x2_prob_total.append(prob_x2[i+gg_sample.argmax()*len(rand_idx)])
				loss.append(F.binary_cross_entropy(prob_x2[i+gg_sample.argmax()*len(rand_idx)], target[sample_idx[i]]))
				loss.append(F.binary_cross_entropy(prob_x1[i+gg_rand.argmax()*len(rand_idx)], target[rand_idx[i]]))  
				loss.append(F.binary_cross_entropy(prob_joint[i+gg_joint.argmax()*len(rand_idx)], joint_label))  
	  
		if opt.aug_type=='vision':
			# v_prob_pos_filter = v_prob_pos * target.unsqueeze(1)
			# v_prob_neg_filter = v_prob_neg * target.unsqueeze(1)     
			# v_pos = v_prob_pos_filter[v_prob_pos_filter!=0]
			# v_neg = v_prob_neg_filter[v_prob_neg_filter!=0]
			# loss =  (-torch.log(v_pos/(v_pos+v_neg))).mean()
			
			return torch.mean(torch.stack(loss))
		elif opt.aug_type=='audio':
			# a_prob_pos_filter = a_prob_pos * target.unsqueeze(1)
			# a_prob_neg_filter = a_prob_neg * target.unsqueeze(1)     
			# a_pos = a_prob_pos_filter[a_prob_pos_filter!=0]
			# a_neg = a_prob_neg_filter[a_prob_neg_filter!=0]
			# loss =  (-torch.log(a_pos/(a_pos+a_neg))).mean()

			return torch.mean(torch.stack(loss))
		elif opt.aug_type=='yybag':
			x1_prob_total_filiter = torch.stack(x1_prob_total)*target[rand_idx]
			x2_prob_total_filiter = torch.stack(x2_prob_total)*target[sample_idx]

			interval = int(len(x2_prob_total_filiter)/3)
			pos_bag = x2_prob_total_filiter[:interval].sum(-1)
			neg_bag = x2_prob_total_filiter[interval:-interval].sum(-1) + x2_prob_total_filiter[-interval:].sum(-1)
			
			loss = (pos_bag/(neg_bag + pos_bag)).mean() + (1 - (neg_bag/(neg_bag + pos_bag)).mean())
			
			return loss
		elif opt.aug_type=='ada':
			
			feature_dic1 = F.normalize(feature_dic1, dim=-1).squeeze(1)
			feature_dic2 = F.normalize(feature_dic2, dim=-1).squeeze(1)

			target_audio = target[0]
			target_vis = target[1]

			# target_audio = torch.cat((target[0],target[0][rand_idx], target[0][sample_idx]), dim=0)
			# target_vis = torch.cat((target[1],target[1][sample_idx], target[1][rand_idx]), dim=0)
			
			# class_dic1 = F.normalize(class_dic1, dim=-1).squeeze(1)
			# class_dic2 = F.normalize(class_dic2, dim=-1).squeeze(1)


			# filiter_a = class_dic1*target_audio.unsqueeze(-1)
			# filiter_a = filiter_a.sum(1)

			# filiter_v = class_dic2*target_vis.unsqueeze(-1)
			# filiter_v = filiter_v.sum(1)

			# target_x1 = torch.cat((target[0],target[0][rand_idx],target[0][sample_idx]), dim=0)
			# target_x2 = torch.cat((target[1],target[1][sample_idx],target[1][rand_idx]), dim=0)

			# corr = torch.mm(filiter_a,filiter_v.permute(1,0))

			# corr.clamp_(max=1)



			corr = torch.mm(target_audio,target_vis.permute(1,0))
			# corr.clamp_(max=1)
			corr[corr!=0] = opt.smooth


			# all_labels = torch.logical_or(target_x1, target_x2).float()
			# corr_norm = corr / all_labels.sum(1)
			# corr_norm[corr_norm!=0] = 1

			sim = torch.mm(feature_dic1, feature_dic2.permute(1,0))
			pos = torch.sum(torch.exp(sim/opt.tmp)*(corr.detach()), dim=1) +1e-10
			neg = torch.sum(torch.exp(sim/opt.tmp)*(1-corr.detach()), dim=1) +1e-10

			
			loss =  (-torch.log(pos/(pos+neg))).mean()

			return loss


		elif opt.aug_type=='mix':
			
			
			# corr = torch.mm(target[sample_idx],target[rand_idx].permute(1,0))
			# corr[corr!=0] = 1

			# corr_copy = torch.clone(corr)
			# corr_copy[corr_copy!=0] = 1

			# exact_same = target[sample_idx].sum(-1)
			# corr[corr ==exact_same]=1

			# norm_feature_1 = F.normalize(feature_dic1, dim=-1)
			# norm_feature_2 = F.normalize(feature_dic2, dim=-1)

			# sim = torch.mm(norm_feature_1, norm_feature_2.permute(1,0))
			

			# # diag = torch.diag(torch.ones(corr.size(0))).to(prob_x2.device)
			# pos = torch.sum(torch.exp(sim/opt.tmp)*(corr), dim=1) +1e-10
			# neg = torch.sum(torch.exp(sim/opt.tmp)*(1-corr_copy), dim=1) 

			# loss =  (-torch.log(pos/(pos+neg))).mean()
			return torch.mean(torch.stack(loss))
			# return loss
		elif opt.aug_type=='mimix':
			
			corr = torch.mm((target[rand_idx] + target[rand_idx]).clamp_(min=0, max=1),(target[rand_idx] + target[rand_idx]).clamp_(min=0, max=1).permute(1,0))
			corr[corr!=0] = 1

			corr_copy = torch.clone(corr)
			corr_copy[corr_copy!=0] = 1

			exact_same = target[sample_idx].sum(-1)
			corr[corr ==exact_same]=1

			norm_feature_1 = F.normalize(x1, dim=-1)
			norm_feature_2 = F.normalize(x2, dim=-1)

			sim = torch.mm(norm_feature_1, norm_feature_2.permute(1,0))
			

			# diag = torch.diag(torch.ones(corr.size(0))).to(prob_x2.device)
			pos = torch.sum(torch.exp(sim/opt.tmp)*(corr), dim=1) +1e-10
			neg = torch.sum(torch.exp(sim/opt.tmp)*(1-corr_copy), dim=1) 

			loss =  (-torch.log(pos/(pos+neg))).mean()
			# return torch.mean(torch.stack(loss))
			return loss

		# v_prob_pos_filter = v_prob_pos * target.unsqueeze(1)
		# v_prob_neg_filter = v_prob_neg * target.unsqueeze(1)     
		# a_prob_pos_filter = a_prob_pos * target.unsqueeze(1)     
		# a_prob_neg_filter = a_prob_neg * target.unsqueeze(1)

		# v_pos = v_prob_pos_filter[v_prob_pos_filter!=0]
		# v_neg = v_prob_neg_filter[v_prob_neg_filter!=0]
		# a_pos = a_prob_pos_filter[a_prob_pos_filter!=0]
		# a_neg = a_prob_neg_filter[a_prob_neg_filter!=0]


		# loss =  (-torch.log(v_pos/(v_pos+v_neg))).mean() + ((-torch.log(a_pos/(a_pos+a_neg)))).mean()

		# return loss 


class YBLoss(torch.nn.Module):
	"""
	Contrastive loss function.
	Based on:
	"""
	def __init__(self, margin=1.0, τ=0.05):
		super(YBLoss, self).__init__()
		self.margin = margin
		self.τ = τ
	def forward(self, all_prob,audio_idx,vis_idx,target,opt):


		v_prob_pos = torch.zeros(all_prob.shape[0]-len(audio_idx),1,25).to(all_prob.device)
		v_prob_neg = torch.zeros(all_prob.shape[0]-len(audio_idx),1,25).to(all_prob.device)
		v_prob_neg_counter = torch.zeros(all_prob.shape[0]-len(audio_idx),1,1).to(all_prob.device)

		a_prob_pos = torch.zeros(all_prob.shape[0]-len(audio_idx),1,25).to(all_prob.device)
		a_prob_neg = torch.zeros(all_prob.shape[0]-len(audio_idx),1,25).to(all_prob.device)
		a_prob_neg_counter = torch.zeros(all_prob.shape[0]-len(audio_idx),1,1).to(all_prob.device)

		for i in range(len(audio_idx)):
			if (target[audio_idx[i]] * target[vis_idx[i]]).sum()==0:
				
				if opt.exp:
					if opt.pos_pool == 'max':
						a_prob_pos[audio_idx[i]] = torch.exp(all_prob[audio_idx[i],:,0,:].max(0)[0])
						v_prob_pos[vis_idx[i]] = torch.exp(all_prob[vis_idx[i],:,1,:].max(0)[0])
					elif opt.pos_pool == 'mean':
						a_prob_pos[audio_idx[i]] = torch.exp(all_prob[audio_idx[i],:,0,:].mean(0))
						v_prob_pos[vis_idx[i]] = torch.exp(all_prob[vis_idx[i],:,1,:].mean(0))

					if opt.neg_pool == 'max':
						a_prob_neg[audio_idx[i]] += torch.exp(all_prob[-len(audio_idx)+i,:,:,:].max(0)[0][1])
						a_prob_neg_counter[audio_idx[i]] += 1
						v_prob_neg[vis_idx[i]] += torch.exp(all_prob[-len(audio_idx)+i,:,:,:].max(0)[0][0])
						v_prob_neg_counter[vis_idx[i]] += 1
					elif opt.neg_pool == 'mean':
						a_prob_neg[audio_idx[i]] += torch.exp(all_prob[-len(audio_idx)+i,:,:,:].mean(0)[1])
						a_prob_neg_counter[audio_idx[i]] += 1
						v_prob_neg[vis_idx[i]] += torch.exp(all_prob[-len(audio_idx)+i,:,:,:].mean(0)[0])
						v_prob_neg_counter[vis_idx[i]] += 1
				else:
					if opt.pos_pool == 'max':
						a_prob_pos[audio_idx[i]] = all_prob[audio_idx[i],:,0,:].max(0)[0]
						v_prob_pos[vis_idx[i]] = all_prob[vis_idx[i],:,1,:].max(0)[0]
					elif opt.pos_pool == 'mean':
						a_prob_pos[audio_idx[i]] = all_prob[audio_idx[i],:,0,:].mean(0)
						v_prob_pos[vis_idx[i]] = all_prob[vis_idx[i],:,1,:].mean(0)

					if opt.neg_pool == 'max':
						a_prob_neg[audio_idx[i]] += all_prob[-len(audio_idx)+i,:,:,:].max(0)[0][1]
						a_prob_neg_counter[audio_idx[i]] += 1
						v_prob_neg[vis_idx[i]] += all_prob[-len(audio_idx)+i,:,:,:].max(0)[0][0]
						v_prob_neg_counter[vis_idx[i]] += 1
					elif opt.neg_pool == 'mean':
						a_prob_neg[audio_idx[i]] += all_prob[-len(audio_idx)+i,:,:,:].mean(0)[1]
						a_prob_neg_counter[audio_idx[i]] += 1
						v_prob_neg[vis_idx[i]] += all_prob[-len(audio_idx)+i,:,:,:].mean(0)[0]
						v_prob_neg_counter[vis_idx[i]] += 1


		
		v_prob_pos_filter = v_prob_pos * target.unsqueeze(1)

		v_prob_neg_filter = v_prob_neg * target.unsqueeze(1)     
		a_prob_pos_filter = a_prob_pos * target.unsqueeze(1)     
		a_prob_neg_filter = a_prob_neg * target.unsqueeze(1)

		v_pos = v_prob_pos_filter[v_prob_pos_filter!=0]
		v_neg = v_prob_neg_filter[v_prob_neg_filter!=0]
		a_pos = a_prob_pos_filter[a_prob_pos_filter!=0]
		a_neg = a_prob_neg_filter[a_prob_neg_filter!=0]


		loss =  (-torch.log(v_pos/(v_pos+v_neg))).mean() + ((-torch.log(a_pos/(a_pos+a_neg)))).mean()

		return loss 
		
class ContrastiveLoss(torch.nn.Module):
	"""
	Contrastive loss function.
	Based on:
	"""

	def __init__(self, margin=1.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def check_type_forward(self, in_types):
		assert len(in_types) == 3

		x0_type, x1_type, y_type = in_types
		assert x0_type.size() == x1_type.shape
		assert x1_type.size()[0] == y_type.shape[0]
		assert x1_type.size()[0] > 0
		assert x0_type.dim() == 2
		assert x1_type.dim() == 2
		assert y_type.dim() == 1

	def forward(self, x0, x1, y):
		self.check_type_forward((x0, x1, y))

		# euclidian distance
		diff = x0 - x1
		dist_sq = torch.sum(torch.pow(diff, 2), 1)
		dist = torch.sqrt(dist_sq)

		mdist = self.margin - dist
		dist = torch.clamp(mdist, min=0.0)
		loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
		loss = torch.sum(loss) / 2.0 / x0.size()[0]
		return loss

class InfoNCELoss(torch.nn.Module):
	"""
	Contrastive loss function.
	Based on:
	"""
	def __init__(self, margin=1.0, τ=0.05):
		super(InfoNCELoss, self).__init__()
		self.margin = margin
		self.τ = τ

	def check_type_forward(self, in_types):
		# assert len(in_types) == 3

		x0_type, x1_type = in_types
		assert x0_type.size() == x1_type.shape
		# assert x1_type.size()[0] == y_type.shape[0]
		# assert x1_type.size()[0] > 0
		# assert x0_type.dim() == 2
		# assert x1_type.dim() == 2
		# assert y_type.dim() == 1

	# def forward(self, x0, x1,eff):
	#     set_trace()
	#     self.check_type_forward((x0, x1))
	#     x0 = x0.view(x0.shape[0],-1)
	#     x1 = x1.view(x0.shape[0],-1)
	#     x0 = F.normalize(x0, p=2, dim=1)
	#     x1 = F.normalize(x1, p=2, dim=1)

		
	#     sim = torch.mm(x0, x1.T)
	#     pos = torch.eye(sim.shape[0]).to(x0.device)
	#     neg = 1-pos
	#     loss = -torch.log(torch.exp((sim*pos).sum(0)/self.τ)/ (torch.exp(sim*neg/self.τ).sum(0)+ 1e-8))  #torch.exp((sim*pos).sum(0)/self.tmp) 
		
	#     return loss.mean()

	def forward(self, q, k):
		# self.check_type_forward((q, k))
		# N is the batch size
		fa = q
		fv = k
		N = q.shape[0]
		
		# C is the dimensionality of the representations
		C = q.shape[1]
		q = q.reshape(q.shape[0]*10,-1)
		k = k.reshape(k.shape[0]*10,-1)

		# v = v.view(k.shape[0],-1)

		q = F.normalize(q, p=2, dim=-1)
		k = F.normalize(k, p=2, dim=-1)
		# v = F.normalize(v, p=2, dim=1)

		
		# bmm stands for batch matrix multiplication
		# If mat1 is a b×n×m tensor, mat2 is a b×m×p tensor, 
		# then output will be a b×n×p tensor. 
		sim = torch.mm(q, k.T)
		
		block = torch.ones(10,10)
		pos_w = torch.eye(sim.shape[0]).to(q.device)

		for i in range(N): # maximize video level sim
			pos_w[(i)*10:(i+1)*10,(i)*10:(i+1)*10] = block

		neg_w = 1-pos_w[:,:sim.shape[1]]

		pos = torch.exp(torch.div(sim,self.τ))*pos_w[:,:sim.shape[1]]
		pos = pos.sum(1)


		##　yanbo aug
		# sim_mix = torch.mm(q, v.T)
		# pos_aug = torch.exp(torch.div(sim_mix,self.τ))*pos_w
		# pos_aug = pos_aug.sum(1)
		
		
		# performs matrix multiplication between query and queue tensors
		neg = torch.sum(torch.exp(torch.div(sim,self.τ))*neg_w, dim=1)
	
		# sum is over positive as well as negative samples
		denominator = neg + pos 
		return torch.mean(-torch.log(torch.div(pos,denominator+1e-8)+1e-8))

class MaskInfoNCELoss(torch.nn.Module):
	"""
	Contrastive loss function.
	Based on:
	"""
	def __init__(self, margin=1.0, τ=0.05):
		super(MaskInfoNCELoss, self).__init__()
		self.margin = margin
		self.τ = τ

	def forward(self, q, k, mask):
		# self.check_type_forward((q, k))
		# N is the batch size
		N = q.shape[0]
		
		# C is the dimensionality of the representations
		C = q.shape[1]
		q = q.view(q.shape[0],-1)
		k = k.view(k.shape[0],-1)
		# v = v.view(k.shape[0],-1)

		q = F.normalize(q, p=2, dim=1)
		k = F.normalize(k, p=2, dim=1)
		# v = F.normalize(v, p=2, dim=1)

		
		# bmm stands for batch matrix multiplication
		# If mat1 is a b×n×m tensor, mat2 is a b×m×p tensor, 
		# then output will be a b×n×p tensor. 
		sim = torch.mm(q, k.T)
		tmp_zeros = torch.zeros((sim.shape[0] - mask.shape[0], sim.shape[1])).to(q.device)
		# if len(tmp_zeros) !=0:
		#     print('neg_work')
		mask_pos = torch.cat((mask, tmp_zeros), dim=0)
		neg_w = 1-mask_pos
		pos = torch.exp(torch.div(sim, self.τ))*mask_pos # + torch.exp(torch.div(sim_vv,self.τ))*(mask-pos_w)
		pos = pos.sum(1)


		##　yanbo aug
		# sim_mix = torch.mm(q, v.T)
		# pos_aug = torch.exp(torch.div(sim_mix,self.τ))*pos_w
		# pos_aug = pos_aug.sum(1)
		
		
		# performs matrix multiplication between query and queue tensors
		neg = torch.sum(torch.exp(torch.div(sim,self.τ))*neg_w, dim=1) #+ torch.sum(torch.exp(torch.div(sim_vv,self.τ))*neg_w, dim=1)
	
		# sum is over positive as well as negative samples
		denominator = neg + pos 
		return torch.mean(-torch.log(torch.div(pos,denominator+1e-8)+1e-8))