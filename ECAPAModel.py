'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from a_softmax import Asoftmax
from l_softmax import AngleLoss
from model import ECAPA_TDNN

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
		super(ECAPAModel, self).__init__()
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).cuda()
		## Classifier
		# self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda()
		self.speaker_loss = AngleLoss().cuda()

		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		i = 0
		for num, (data, labels) in enumerate(loader, start = 1):
			self.zero_grad()
			labels            = torch.LongTensor(labels).cuda()
			speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug = True)
			nloss, prec       = self.speaker_loss.forward(speaker_embedding, labels)			
			nloss.backward()
			self.optim.step()
			index += len(labels)
			top1 += prec
			loss += nloss.detach().cpu().numpy()
			if i % 200 == 0:
				sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
				" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
				" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
				sys.stderr.flush()
				sys.stdout.write("\n")
			i += 1
		return loss/num, lr, top1/index*len(labels)

	def eval_network(self, eval_list, eval_path, flag=False):
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats).cuda()
			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

		for line in lines:			
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))

		# scores 是分数 lable 1 是相同   0是不同
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, minDCFThresholds = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
		if flag:
			print(scores, "socres!!!!!")
			print('\nfnrs', fnrs, '\nfprs', fprs)
			print('thresholds=', thresholds, 'minDCF=', minDCF, 'minDCFThresholds=', minDCFThresholds)
			self.get_table(scores, labels)
		return EER, minDCF

	# 获取图表
	def get_table(self, scores=[], labels=[]):
		far_list = []
		frr_list = []
		# 放松模式
		flag = True
		dcf_2_list = []
		dcf_1_list = []
		for th in range(-1, 101):
			fail_reject_count = 0
			fail_accept_count = 0
			reject_count = 0
			accept_count = 0
			for j in range(0, len(scores)):
				score = scores[j]*100
				if labels[j]==0:
					reject_count+=1
					if score >= th:
						fail_accept_count += 1
				else:
					accept_count+=1
					if score < th:
						fail_reject_count += 1
			far = fail_accept_count/reject_count
			far_list.append(far)
			frr = fail_reject_count/accept_count
			frr_list.append(frr)
			cost = 0.99*frr + 0.01*far
			dcf_1_list.append(cost)
			cost = 0.01*frr + 0.99*far
			dcf_2_list.append(cost)
		print('\n far_list \n')
		print(far_list)
		print('\n frr_list \n')
		print(frr_list)
		print('\n dcf_1_list \n')
		print(dcf_1_list)
		print('\n dcf_2_list \n')
		print(dcf_2_list)
		print('\n')
		return far_list, frr_list, dcf_1_list, dcf_2_list


	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)