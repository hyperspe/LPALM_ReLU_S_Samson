import torch
import torch.nn as nn
import numpy as np
from ISTA_LLT import ISTA_LLT
from LISTA import LISTA
from LISTA_CP import LISTA_CP
from LPALM import LPALM
import matplotlib.pyplot as plt
from scipy.io import savemat

def train_non_blind(train_loader, val_loader, num_epochs=10, T=10, alpha = 10, mode = None,L_shared = True, theta_shared=True,We_shared=True,G_shared=True,W_shared=True,noise_A=None):

	criterion = nn.MSELoss()
	layers = np.arange(T)
	A = next(iter(train_loader))[1]
	
	if mode == 'LC':
		model = LISTA(T, alpha= alpha, A=A,theta_shared=theta_shared,We_shared=We_shared,G_shared= G_shared)
		
	elif mode == 'CP':
		model = LISTA_CP(T,alpha=alpha,A=A,theta_shared=theta_shared, W_shared= W_shared)
		
	elif mode == 'LLT':
		model = ISTA_LLT(T,L_shared=L_shared, theta_shared= theta_shared)
		
	else: 
		print("The entered mode doesn't correspond to a model")
	
	total_params = [p.numel() for p in model.parameters()]
	print(total_params)

	optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0001, betas=(0.9, 0.999))

	train_total_loss = []
	val_total_loss = []
	
	list_err_val_layer = [[0 for i in range(T)] for j in range(num_epochs)]
	lmda = 1.5e-6
	
	for epoch in range(num_epochs):
		train_total = 0
		model.train()
		for i, (X,A,S) in enumerate(train_loader):

			optimizer.zero_grad()
			if mode == 'LC':
				S_pred, S_s = model(X)
			else:
				S_pred, S_s = model(X,A,noise_A=noise_A)
			train_loss = torch.numel(S) * criterion(S.float(), S_pred.float()) / torch.norm(S.float())**2
			train_loss.backward()
			optimizer.step()
			train_total += train_loss.item()
		train_total /= i+1
		train_total_loss.append(train_total)

		model.eval()
		with torch.no_grad():
			val_total = 0
			for i, (X,A,S) in enumerate(val_loader):
				if mode == 'LC':
					S_pred, S_s_val = model(X)
				else:
					S_pred,S_s_val = model(X,A,noise_A=noise_A)
				val_loss = torch.numel(S) *  criterion(S.float(), S_pred.float()) / torch.norm(S.float())**2
				val_total += val_loss.item()
				for j,s in enumerate(S_s_val):
					list_err_val_layer[epoch][j] += torch.numel(S) *criterion(S.float(), s.float()).item() / torch.norm(S.float())**2
			list_err_val_layer[epoch] = [x/(i+1) for x in list_err_val_layer[epoch]]
			val_total /= i+1
			val_total_loss.append(val_total)
		
		print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
		if epoch % 5 == 0:
			print()

		print("epoch:{} | training loss:{:.5f} | validation loss:{:.5f} ".format(epoch, train_total,val_total))
		
		torch.save(model, mode +str(epoch)+'.pth')

	return train_total_loss, val_total_loss,model

def train_blind(train_loader, val_loader, num_epochs=100, T=25,learn_L_S = False ,L_S_shared=False, LISTA_S = False, W_X_S_shared= False, W_S_S_shared=False,theta_shared=False,  LISTA_CP_S = False, W_CP_S_shared= True, ISTA_LLT_S = False, learn_L_A = False , L_A_shared= True, LISTA_A = False , W_A_A_shared = True, W_X_A_shared = True, LISTA_CP_A = False, W_CP_A_shared = True,non_update_A = False):
	
	torch.autograd.set_detect_anomaly(True)
	criterion = nn.MSELoss()
	
	A = next(iter(train_loader))[1]
	S = next(iter(train_loader))[2]
	model = LPALM(T, S=S , A=A,learn_L_S = learn_L_S ,L_S_shared=L_S_shared, LISTA_S = LISTA_S, W_X_S_shared= W_X_S_shared , W_S_S_shared= W_S_S_shared ,theta_shared= theta_shared,  LISTA_CP_S = LISTA_CP_S, W_CP_S_shared= W_CP_S_shared , ISTA_LLT_S = ISTA_LLT_S, learn_L_A = learn_L_A , L_A_shared= L_A_shared , LISTA_A = LISTA_A , W_A_A_shared = W_A_A_shared, W_X_A_shared = W_X_A_shared , LISTA_CP_A = LISTA_CP_A, W_CP_A_shared = W_CP_A_shared, non_update_A = non_update_A)
	
	total_params = [p.numel() for p in model.parameters()]
	print(total_params)
	
	optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0001, betas=(0.9, 0.999))

	train_total_loss = []
	val_total_loss = []
################################################################################
	list_err_train_S_layer = [[0 for i in range(T)] for j in range(num_epochs)]#
	list_err_train_A_layer = [[0 for i in range(T)] for j in range(num_epochs)]#
################################################################################
	list_err_val_S_layer = [[0 for i in range(T)] for j in range(num_epochs)]
	list_err_val_A_layer = [[0 for i in range(T)] for j in range(num_epochs)]
	
	for epoch in range(num_epochs):
		train_total = 0
		model.train()
		for i, (X,A,S) in enumerate(train_loader):
			optimizer.zero_grad()
			S_pred, A_pred, S_s_train, A_s_train = model(X)
			train_loss=0
			train_loss = torch.numel(S) * criterion(S.float(), S_pred.float()) / torch.norm(S.float())**2 + torch.numel(A) * criterion(A.float(),A_pred.float()) / torch.norm(A.float())**2
			train_loss.backward()
			optimizer.step()
			train_total += train_loss.item()
##Enregistrement des NMSE train pour A et S ##################################################################
			for j,s in enumerate(S_s_train):
				list_err_train_S_layer[epoch][j] += torch.numel(S) * criterion(S.float(), s.float()).item() / torch.norm(S.float())**2
			for j,a in enumerate(A_s_train):
				list_err_train_A_layer[epoch][j] += torch.numel(A) * criterion(A.float(), a.float()).item() / torch.norm(A.float())**2
					
		list_err_train_S_layer[epoch] = [x/(i+1) for x in list_err_train_S_layer[epoch]]
		list_err_train_A_layer[epoch] = [x/(i+1) for x in list_err_train_A_layer[epoch]]
###############################################################################################################
		train_total /= i+1
		train_total_loss.append(train_total)
		
		model.eval()
		with torch.no_grad():
			val_total = 0
			for i, (X,A,S) in enumerate(val_loader):
				S_pred, A_pred, S_s_val, A_s_val = model(X)
				val_loss = torch.numel(S) * criterion(S.float(), S_pred.float()) / torch.norm(S.float())**2 + torch.numel(A) * criterion(A.float(),A_pred.float()) / torch.norm(A.float())**2
				val_total += val_loss.item()
				for j,s in enumerate(S_s_val):
					list_err_val_S_layer[epoch][j] += torch.numel(S) * criterion(S.float(), s.float()).item() / torch.norm(S.float())**2
				for j,a in enumerate(A_s_val):
					list_err_val_A_layer[epoch][j] += torch.numel(A) * criterion(A.float(), a.float()).item() / torch.norm(A.float())**2
					
			list_err_val_S_layer[epoch] = [x/(i+1) for x in list_err_val_S_layer[epoch]]
			list_err_val_A_layer[epoch] = [x/(i+1) for x in list_err_val_A_layer[epoch]]
			val_total /= i+1
			val_total_loss.append(val_total)

		
		print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
		if epoch % 5 == 0:
			print()

		print("epoch:{} | training loss:{:.5f} | validation loss:{:.5f} ".format(epoch, train_total,val_total))
		torch.save(model,'LPALM'+str(epoch)+'.pth')
		
	min_val_loss_idx = val_total_loss.index(min(val_total_loss))
	layers = np.arange(T)
	
	mdic = {"layers": layers , "NMSE_val_S": list_err_val_S_layer[min_val_loss_idx]}
	savemat("NMSE_val_S.mat", mdic)
	
	mdic = {"layers": layers , "NMSE_val_A": list_err_val_A_layer[min_val_loss_idx]}
	savemat("NMSE_val_A.mat", mdic)
	
    
	#Ajout des NMS train pour A et S (affichage + sauvegarde)#############################
	mdic = {"layers": layers , "NMSE_train_S": list_err_train_S_layer[min_val_loss_idx]}
	savemat("NMSE_train_S.mat", mdic)
	
	mdic = {"layers": layers , "NMSE_train_A": list_err_train_A_layer[min_val_loss_idx]}
	savemat("NMSE_train_A.mat", mdic)
	######################################################################################
    
	#Sauvegarde de la training_loss##############
	training_loss = np.array(train_total_loss)  #
	np.savez("training_loss", training_loss)    #
    #Sauvegarde de la validation_loss############
	validation_loss = np.array(val_total_loss)  #
	np.savez("validation_loss", validation_loss)#
	#############################################
		
	plt.plot(list_err_train_S_layer[min_val_loss_idx],label='NMSE S train')
	plt.plot(list_err_val_S_layer[min_val_loss_idx],label='NMSE S val')
	plt.plot(list_err_train_A_layer[min_val_loss_idx],label='NMSE A train')
	plt.plot(list_err_val_A_layer[min_val_loss_idx],label='NMSE A val')
    
	plt.legend()
	plt.xlabel('layer')
	plt.ylabel('NMSE')
	plt.yscale('log')
	plt.show()
	
	return train_total_loss, val_total_loss,model
