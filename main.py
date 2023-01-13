import torch
from torch.utils.data import DataLoader,Dataset,Subset
import matplotlib.pyplot as plt
import matplotlib
import pylab
from utils import My_dataset,train_val_dataset,split_A,plot_train_test_loss,apply_LPALM_realistic
from train import train_non_blind,train_blind
from ISTA import ISTA
from PALM import PALM
from LISTA import LISTA
import argparse
import numpy as np
import torch.nn as nn
import pickle

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--ISTA', action= 'store_true', required=False)
parser.add_argument('--add_noise', action= 'store_true', required=False) 
parser.add_argument('--LISTA_CP',action = 'store_true', required=False)
parser.add_argument('--LISTA',action = 'store_true', required=False)
parser.add_argument('--ISTA_LLT',action = 'store_true', required=False)
parser.add_argument('--LPALM', action = 'store_true', required = False)
parser.add_argument('--shared_L', action='store_true', required= False) 
parser.add_argument('--share_theta',action = 'store_true' ,required = False)
parser.add_argument('--share_We', action = 'store_true',required=False)
parser.add_argument('--share_G', action = 'store_true',required = False)
parser.add_argument('--share_W_CP', action= 'store_true',required= False)
parser.add_argument('--learn_L_S', action= 'store_true', required= False)
parser.add_argument('--L_S_shared', action= 'store_true', required= False)
parser.add_argument('--LISTA_S', action= 'store_true', required= False)
parser.add_argument('--W_X_S_shared', action= 'store_true', required= False)
parser.add_argument('--W_S_S_shared', action= 'store_true', required= False)
parser.add_argument('--theta_shared', action= 'store_true', required= False)
parser.add_argument('--LISTA_CP_S', action= 'store_true', required= False)
parser.add_argument('--W_CP_S_shared', action= 'store_true', required= False)
parser.add_argument('--ISTA_LLT_S',action='store_true', required=False)
parser.add_argument('--learn_L_A', action= 'store_true', required= False)
parser.add_argument('--L_A_shared', action= 'store_true', required= False)
parser.add_argument('--LISTA_A', action= 'store_true', required= False)
parser.add_argument('--W_A_A_shared', action= 'store_true', required= False)
parser.add_argument('--W_X_A_shared', action= 'store_true', required= False)
parser.add_argument('--LISTA_CP_A', action= 'store_true', required= False)
parser.add_argument('--W_CP_A_shared', action= 'store_true', required= False)
parser.add_argument('--non_update_A',action='store_true',required=False)
parser.add_argument('--add_noise_A', action= 'store_true', required= False)

args = parser.parse_args()
#dataset = My_dataset(beta = 0.3 , t = 500, noise = args.add_noise, normalize_S_lines=True,synthetic=True)
#with open('dataset_4_sources.pickle', 'wb') as f:
#	pickle.dump(dataset, f)

#with open('dataset_4_sources.pickle', 'rb') as f: #=> Jeu de données original
#with open('jeu200_10000.pickle', 'rb') as f:
#      dataset = pickle.load(f)
#datasets = split_A(dataset)
#train_set = datasets['train']
#val_set = datasets['val']
#torch.manual_seed(42)
#train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0) #4)
#val_loader = DataLoader(val_set,batch_size=1, shuffle=False, num_workers=0) #4)

#[Already splitted dataset]
with open('samson_train_set.pickle', 'rb') as f:
      train_set = pickle.load(f)
with open('samson_val_set.pickle', 'rb') as f:
      val_set = pickle.load(f)
torch.manual_seed(42)
train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0) #4)
val_loader = DataLoader(val_set,batch_size=1, shuffle=False, num_workers=0) #4)


#apply_LPALM_realistic(val_loader)
if args.ISTA:
	lmda = 1.1e-5
	nb_iter=25
	if args.add_noise_A:
		noise_A = 10
	else:
		noise_A = None
	S_est , loss, loss_per_layer = ISTA(val_set,lmda,nb_iter,nb_iter_fixed = True,noise_A=noise_A)

if args.ISTA_LLT:
	
	alpha =1e-5
	mode = 'LLT'
	if args.add_noise_A:
		noise_A = 10
	else:
		noise_A = None
	train_total_loss, test_total_loss, model = train_non_blind(train_loader, val_loader, num_epochs=100, T= 25 , alpha = alpha, mode=mode, L_shared = args.shared_L, theta_shared=args.share_theta,We_shared=args.share_We,G_shared=args.share_G ,W_shared=args.share_W_CP ,noise_A=noise_A)
	plot_train_test_loss(train_total_loss,test_total_loss)

	
if args.LISTA_CP:
	alpha = 1e-5
	mode = 'CP'
	if args.add_noise_A:
		noise_A = 10
	else:
		noise_A = None
	train_total_loss, test_total_loss,model = train_non_blind(train_loader, val_loader, num_epochs=100, T= 25 , alpha = alpha,mode=mode,L_shared = args.shared_L, theta_shared=args.share_theta,We_shared=args.share_We,G_shared=args.share_G ,W_shared=args.share_W_CP ,noise_A=noise_A)
	plot_train_test_loss(train_total_loss,test_total_loss)
	
if args.LISTA:
	alpha= 1e-5
	mode = 'LC'
	print(args.share_theta)
	print(args.share_We)
	print(args.share_G)
	train_total_loss, test_total_loss, model = train_non_blind(train_loader, val_loader, num_epochs=10, T= 25 , alpha = alpha,mode=mode,L_shared = args.shared_L, theta_shared=args.share_theta,We_shared=args.share_We,G_shared=args.share_G ,W_shared=args.share_W_CP ,noise_A=None)
	plot_train_test_loss(train_total_loss,test_total_loss)


if args.LPALM:

	train_total_loss, test_total_loss, model = train_blind(train_loader, val_loader, num_epochs=100, T= 25, learn_L_S = args.learn_L_S, L_S_shared=args.L_S_shared, LISTA_S = args.LISTA_S, W_X_S_shared= args.W_X_S_shared , W_S_S_shared= args.W_S_S_shared ,theta_shared= args.theta_shared,  LISTA_CP_S = args.LISTA_CP_S, W_CP_S_shared= args.W_CP_S_shared ,ISTA_LLT_S = args.ISTA_LLT_S, learn_L_A = args.learn_L_A , L_A_shared= args.L_A_shared , LISTA_A = args.LISTA_A , W_A_A_shared = args.W_A_A_shared, W_X_A_shared = args.W_X_A_shared , LISTA_CP_A = args.LISTA_CP_A, W_CP_A_shared = args.W_CP_A_shared,non_update_A= args.non_update_A)
	plot_train_test_loss(train_total_loss,test_total_loss)
