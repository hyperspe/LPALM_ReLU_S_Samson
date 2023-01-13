import torch
import torch.nn as nn
import numpy as np
from utils import prox_theta,tile,prox

import scipy.io

#Chargement des estimations issues de SNPA sur Samson
mat = scipy.io.loadmat('A_snpa')
data = list(mat.items())
an_array = np.array(data,dtype=object)
A_snpa = an_array[3,1]
A_snpa = torch.tensor(np.reshape(A_snpa,(1,156,3)))

mat = scipy.io.loadmat('S_snpa')
data = list(mat.items())
an_array = np.array(data,dtype=object)
S_snpa = an_array[3,1]
S_snpa = torch.tensor(np.reshape(S_snpa,(1,3,9025)))




class LPALM(nn.Module):
	def __init__(self, T=25,S= None,A=None,learn_L_S = False ,L_S_shared=False, LISTA_S = False, W_X_S_shared=False, W_S_S_shared=False, theta_shared=False, LISTA_CP_S=True, W_CP_S_shared= False, ISTA_LLT_S = False, learn_L_A = True , L_A_shared=False, LISTA_A = False , W_A_A_shared =False, W_X_A_shared =False, LISTA_CP_A = False, W_CP_A_shared = False, non_update_A = False):
		super(LPALM, self).__init__()
		
		
		torch.set_default_tensor_type(torch.DoubleTensor)
		self.T = T
		self.S = S
		self.A = A
		#######################################################     S
		self.learn_L_S = learn_L_S
		self.L_S_shared = L_S_shared
		
		self.LISTA_S = LISTA_S
		self.W_X_S_shared = W_X_S_shared
		self.W_S_S_shared = W_S_S_shared
		self.theta_shared = theta_shared
		
		self.LISTA_CP_S = LISTA_CP_S
		self.W_CP_S_shared = W_CP_S_shared
		
		self.ISTA_LLT_S = ISTA_LLT_S
		#######################################################     A
		self.learn_L_A = learn_L_A
		self.L_A_shared = L_A_shared
		
		self.LISTA_A = LISTA_A
		self.W_A_A_shared = W_A_A_shared
		self.W_X_A_shared = W_X_A_shared
		
		self.LISTA_CP_A = LISTA_CP_A
		self.W_CP_A_shared = W_CP_A_shared
		
		self.non_update_A = non_update_A
		
		##########
		self.alpha =0.00001
		self.L_S = torch.tensor((1.001) * np.linalg.norm(self.A[0].numpy(),ord=2)**2)
		self.L_A = torch.tensor((1.001) * np.linalg.norm(self.S[0].numpy(),ord=2)**2)
		
		#############################################################################
		
		if self.learn_L_S:
			if not self.L_S_shared:
				self.L_S = self.L_S.repeat(self.T,1)
			self.L_S =nn.Parameter(self.L_S,requires_grad= True)
		
		if self.LISTA_S:
			self.theta = (self.alpha / self.L_S).clone().detach()
			self.W_X_S = torch.div(torch.transpose(self.A[0],0,1), self.L_S)
			self.W_S_S = torch.eye(self.A.shape[2]) - torch.matmul(torch.div(torch.transpose(self.A[0],0,1), self.L_S), self.A[0])
			
			if not self.theta_shared:
				self.theta = self.theta.repeat(self.T,1)
			self.theta = nn.Parameter(self.theta, requires_grad= True)
			
			if not self.W_X_S_shared:
				self.W_X_S = self.W_X_S.repeat(self.T,1,1)
			self.W_X_S = nn.Parameter(self.W_X_S, requires_grad= True)
			
			if not self.W_S_S_shared:
				self.W_S_S = self.W_S_S.repeat(self.T,1,1)
			self.W_S_S = nn.Parameter(self.W_S_S, requires_grad= True)
		
		if self.LISTA_CP_S:
			self.theta = (self.alpha / self.L_S).clone().detach()			
			self.We = torch.div(torch.transpose(self.A[0],0,1), self.L_S)
			self.W_CP_S = torch.transpose(self.We,0,1)
			
			if not self.theta_shared:
				self.theta = self.theta.repeat(self.T,1)
			self.theta = nn.Parameter(self.theta, requires_grad= True)
			
			if not self.W_CP_S_shared:
				self.W_CP_S = self.W_CP_S.repeat(self.T,1,1)
			self.W_CP_S = nn.Parameter(self.W_CP_S, requires_grad= True)
			
		if self.ISTA_LLT_S:
			self.theta = (self.alpha/ self.L_S).clone().detach()
			self.theta = self.theta.repeat(self.T,1)
			self.theta = nn.Parameter(self.theta, requires_grad= True)
			self.L_S = self.L_S.repeat(self.T,1)
			self.L_S = nn.Parameter(self.L_S,requires_grad= True)
				
#	############################################################################
		
		if self.learn_L_A:
			if not self.L_A_shared:
				self.L_A = self.L_A.repeat(self.T,1)
			self.L_A =nn.Parameter(self.L_A,requires_grad= True)

		if self.LISTA_A:
			self.W_A_A = torch.eye(self.S.shape[1]) - torch.matmul(self.S[0], torch.div(torch.transpose(self.S[0],0,1), self.L_A))
			self.W_X_A = torch.div(torch.transpose(self.S[0],0,1), self.L_A)
			
			if not self.W_A_A_shared:
				self.W_A_A = self.W_A_A.repeat(self.T,1,1)
			self.W_A_A = nn.Parameter(self.W_A_A,requires_grad=True)
			
			if not self.W_X_A_shared:
				self.W_X_A = self.W_X_A.repeat(self.T,1,1)
			self.W_X_A = nn.Parameter(self.W_X_A,requires_grad=True)	

		if self.LISTA_CP_A:
			self.W_CP_A = torch.div(torch.transpose(self.S[0],0,1), self.L_A)
			
			if not self.W_CP_A_shared:
				self.W_CP_A = self.W_CP_A.repeat(self.T,1,1)
			self.W_CP_A = nn.Parameter(self.W_CP_A,requires_grad=True)
		
		
	def forward(self, X):
	
		b_size = X.shape[0]
        
        
		#initialize S
		S_pred = S_snpa.clone() # self.S #On est censés pouvoir rester assez général        
        
#		S_pred = torch.zeros(b_size,self.A.shape[2], X.shape[2])
#		S_pred = S_pred.type(torch.DoubleTensor)
        
        
		#initialize A
		A_pred = A_snpa.clone() # self.A #A_snpa.clone()        
        
#		A_pred = torch.ones([b_size, X.shape[1], self.S.shape[1]])
#		n_ = torch.norm(A_pred, dim=1)
#		A_pred = A_pred/tile(n_,0,A_pred.shape[1]).reshape(A_pred.shape[0],A_pred.shape[1],A_pred.shape[2])
#		A_pred = A_pred.type(torch.DoubleTensor)
        
        
		S_s= []
		A_s= []
		
		for t in range(self.T):
			

			# iteration on S
			if self.learn_L_S:
				if self.L_S_shared:	
					S_pred = prox(S_pred - (1/self.L_S)*torch.bmm(torch.transpose(A_pred,1,2), (torch.bmm(A_pred,S_pred)-X)), self.alpha, self.L_S)
				else:
					S_pred = prox(S_pred - (1/self.L_S[t])*torch.bmm(torch.transpose(A_pred,1,2), (torch.bmm(A_pred,S_pred)-X)), self.alpha, self.L_S[t])
		
		
			elif self.LISTA_S:
				if self.W_X_S_shared:
					p1 = torch.bmm(self.W_X_S.repeat(b_size,1,1),X)
				else:
					p1 = torch.bmm(self.W_X_S[t].repeat(b_size,1,1),X)
				if self.W_S_S_shared:
					p2 = torch.bmm(self.W_S_S.repeat(b_size,1,1),S_pred)
				else:
					p2 = torch.bmm(self.W_S_S[t].repeat(b_size,1,1),S_pred)
				p3 = p1+p2
				if self.theta_shared:
					S_pred = prox_theta(p3,self.theta)
				else:
					S_pred = prox_theta(p3,self.theta[t])
					
					
##############################################################################
			elif self.LISTA_CP_S:
				if self.W_CP_S_shared:
					S_pred = S_pred + torch.bmm(torch.transpose(self.W_CP_S.repeat(b_size,1,1),1,2),X-torch.bmm(A_pred,S_pred))
				else:
					S_pred = S_pred + torch.bmm(torch.transpose(self.W_CP_S[t].repeat(b_size,1,1),1,2),X-torch.bmm(A_pred,S_pred))
				if self.theta_shared:
#					S_pred = prox_theta(S_pred,self.theta) 
					S_pred = nn.ReLU()(S_pred-self.theta) #If shared, the same between layers, no need for specifying [t]
#					S_pred = nn.Softmax(dim=0)(S_pred)

				else:
#					S_pred = prox_theta(S_pred,self.theta[t]) #Recall t goes until T=25, which is the number of layers.
					S_pred = nn.ReLU()(S_pred-self.theta[t]) 
#					S_pred = nn.Softmax(dim=0)(S_pred)
##############################################################################
					
					
			elif self.ISTA_LLT_S:
				S_pred = prox_theta(S_pred - (1/self.L_S[t])*torch.bmm(torch.transpose(A_pred,1,2), (torch.bmm(A_pred,S_pred)-X)), self.theta[t])
			
            
            
		#############################################################################         
#			#Normalisation des colonnes de S            
#			S_pred = nn.functional.normalize(S_pred, p=1.0, dim=1, eps=1e-12, out=None)
		#############################################################################
    
    
			#iteration on A
			
			if self.learn_L_A:
				p1 = torch.bmm(torch.bmm(A_pred,S_pred)-X,torch.transpose(S_pred,1,2))
				if self.L_A_shared:
					p2 = torch.div(p1, self.L_A)
				else:
					p2 = torch.div(p1, self.L_A[t])
				A_pred = A_pred - p2
					
			elif self.LISTA_A:
				if self.W_A_A_shared:
					p1 = torch.bmm(A_pred,self.W_A_A.repeat(b_size,1,1))
				else:
					p1 = torch.bmm(A_pred,self.W_A_A[t].repeat(b_size,1,1))
				if self.W_X_A_shared:
					p2 = torch.bmm(X,self.W_X_A.repeat(b_size,1,1))
				else:
					p2 = torch.bmm(X,self.W_X_A[t].repeat(b_size,1,1)) 
				A_pred = p1+p2
				
			elif self.LISTA_CP_A:
				p = torch.bmm(A_pred,S_pred) - X
				if self.W_CP_A_shared:
					A_pred = A_pred - torch.bmm(p,self.W_CP_A.repeat(b_size,1,1))
				else:
					A_pred = A_pred -torch.bmm(p,self.W_CP_A[t].repeat(b_size,1,1))
					
			elif self.non_update_A:
				p1 = torch.bmm(torch.bmm(A_pred,S_pred)-X,torch.transpose(S_pred,1,2))
				L_A = torch.tensor((1.001) * np.linalg.norm(S_pred[0].detach().numpy(),ord=2)**2)
				p2 = torch.div(p1, L_A)
				A_pred = A_pred - p2			
			
			#projection on the non-negative orthant		
			for i in range(A_pred.shape[0]):
				A_pred[i] = A_pred[i]*(A_pred[i]>0)
			
			#projection on the unit ball		
			for i in range(A_pred.shape[0]):
				for j in range(A_pred.shape[2]):
					if(torch.any(torch.norm(A_pred[i][:,j])>torch.tensor([1.]))): #Normalisation partielle
						A_pred[i][:,j] /= torch.norm(A_pred[i][:,j].clone())
						
		#############################################################################

			A_s.append(A_pred)
			S_s.append(S_pred)

		return S_pred, A_pred , S_s , A_s
