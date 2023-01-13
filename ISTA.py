import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from utils import prox

def ISTA(set_,lmda,nb_iter,nb_iter_fixed = False,noise_A=None):
	eps =0.01
	criterion = nn.MSELoss()
	S_est = torch.zeros([len(set_), next(iter(set_))[2].shape[0],next(iter(set_))[2].shape[1]], dtype=torch.double) #initialization
	S_est_prev = deepcopy(S_est)
	loss = []
	S_est_list = []
	loss_per_layer = torch.zeros((25))
	it=0
	total_it = 0
	for i, (X,A,S) in enumerate(set_):
		X = torch.from_numpy(X)
		A = torch.from_numpy(A)
		if noise_A is not None:	
			N = np.random.randn(A.shape[0],A.shape[1])
			N = 10.**(-noise_A/20.)*np.linalg.norm(A)/np.linalg.norm(N)*N
			A = A + N
		S = torch.from_numpy(S)
		L = torch.tensor((1+eps) * np.linalg.norm(A.numpy(),ord=2)**2)
		print("example %d",i)
		it= 0 
		while(torch.norm(S_est[i]-S_est_prev[i], p = 'fro') > 1e-6 or it < 2):
			if it>0:
				S_est_prev[i] = deepcopy(S_est[i])
			S_est[i] = prox(S_est[i] - (1 / L) * torch.matmul(torch.transpose(A, 0, 1), (torch.matmul(A, S_est[i]) - X)), lmda, L)
			S_est_list.append(torch.empty_like(S_est[i]).copy_(S_est[i]))
			it += 1
			total_it +=1
			if nb_iter_fixed:
				if it == nb_iter:
					break
		loss.append(torch.numel(S) *criterion(S_est[i],S) / torch.norm(S)**2)
		for j in range(25):
			loss_per_layer[j] += torch.numel(S) *criterion(S_est_list[j],S) / torch.norm(S)**2

	for j in range(25):
		loss_per_layer[j] = loss_per_layer[j]/(i+1)

	print('the average number of iterations required to convergence is: ',total_it/(i+1))
	print('The average loss of ISTA is {:.5f}'.format(sum(loss)/(i+1)))
	return S_est , sum(loss)/(i+1), loss_per_layer
