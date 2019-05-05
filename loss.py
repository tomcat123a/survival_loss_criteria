# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:47:42 2019

@author: Administrator
"""
os.chdir('C:\surv_deep\pysurvival-master\pysurvival-master\pysurvival')
from __future__ import absolute_import
from pysurvival import utils


def get_data(cancer_type):
    X=pd.read_csv('C:/python_study/surviv/survivdeep/'+cancer_type+'_x.csv').values
    YTIME=pd.read_csv('C:/python_study/surviv/survivdeep/'+cancer_type+'_t.csv').values
    YEVENT=pd.read_csv('C:/python_study/surviv/survivdeep/'+cancer_type+'_c.csv').values
    # censored(0) or not(1)
    amatrix=pd.read_csv('C:/python_study/surviv/survivdeep/'+cancer_type+'_w.csv',index_col=False)
    X = torch.from_numpy(x).type(dtype)
	YTIME = torch.from_numpy(ytime).type(dtype)
	YEVENT = torch.from_numpy(yevent).type(dtype)
    return X,YTIME,YEVENT,amatrix


def R_set(x):
	'''Create an indicator matrix of risk sets, where T_j >= T_i.
	Note that the input data have been sorted in descending order.
	Input:
		x: a PyTorch tensor that the number of rows is equal to the number of samples.
	Output:
		indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
	'''
	n_sample = x.size(0)
	matrix_ones = torch.ones(n_sample, n_sample)
	indicator_matrix = torch.tril(matrix_ones)

	return(indicator_matrix)


def neg_par_log_likelihood(pred, ytime, yevent):#event=0,censored
    #ytime should be sorted with increasing order
	'''Calculate the average Cox negative partial log-likelihood.
	Input:
		pred: linear predictors from trained model.
		ytime: true survival time from load_data().
		yevent: true censoring status from load_data().
	Output:
		cost: the cost that is to be minimized.
	'''
	n_observed = yevent.sum(0)
	ytime_indicator = R_set(ytime)
	###if gpu is being used
	if torch.cuda.is_available():
		ytime_indicator = ytime_indicator.cuda()
	###
	risk_set_sum = ytime_indicator.mm(torch.exp(pred)) 
	diff = pred - torch.log(risk_set_sum)
	sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
	cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))

	return(cost)


def c_index(pred, ytime, yevent):
	'''Calculate concordance index to evaluate models.
	Input:
		pred: linear predictors from trained model.
		ytime: true survival time from load_data().
		yevent: true censoring status from load_data().
	Output:
		concordance_index: c-index (between 0 and 1).
	'''
	n_sample = len(ytime)
	ytime_indicator = R_set(ytime)
	ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
	###T_i is uncensored
	censor_idx = (yevent == 0).nonzero()
	zeros = torch.zeros(n_sample)
	ytime_matrix[censor_idx, :] = zeros
	###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
	pred_matrix = torch.zeros_like(ytime_matrix)
	for j in range(n_sample):
		for i in range(n_sample):
			if pred[i] < pred[j]:
				pred_matrix[j, i]  = 1
			elif pred[i] == pred[j]: 
				pred_matrix[j, i] = 0.5
	
	concord_matrix = pred_matrix.mul(ytime_matrix)
	###numerator
	concord = torch.sum(concord_matrix)
	###denominator
	epsilon = torch.sum(ytime_matrix)
	###c-index = numerator/denominator
	concordance_index = torch.div(concord, epsilon)
	###if gpu is being used
	if torch.cuda.is_available():
		concordance_index = concordance_index.cuda()
	###
	return(concordance_index)




net = Cox_PASNet(In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, pathway_mask)
	###if gpu is being used
if torch.cuda.is_available():
	net.cuda()
	###
	###optimizer
opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay = L2)

for epoch in range(Num_Epochs+1):
	net.train()
	opt.zero_grad() ###reset gradients to zeros
	###Randomize dropout masks
	net.do_m1 = dropout_mask(Pathway_Nodes, Dropout_Rate[0])
	net.do_m2 = dropout_mask(Hidden_Nodes, Dropout_Rate[1])
	pred = net(train_x, train_age) ###Forward
	loss = neg_par_log_likelihood(pred, train_ytime, train_yevent) ###calculate loss
	loss.backward() ###calculate gradients
	opt.step() ###update weights and biases

	net.sc1.weight.data = net.sc1.weight.data.mul(net.pathway_mask) ###force the connections between gene layer and pathway layer

		###obtain the small sub-network's connections
	do_m1_grad = copy.deepcopy(net.sc2.weight._grad.data)
	do_m2_grad = copy.deepcopy(net.sc3.weight._grad.data)
	do_m1_grad_mask = torch.where(do_m1_grad == 0, do_m1_grad, torch.ones_like(do_m1_grad))
	do_m2_grad_mask = torch.where(do_m2_grad == 0, do_m2_grad, torch.ones_like(do_m2_grad))
	###copy the weights
	net_sc2_weight = copy.deepcopy(net.sc2.weight.data)
	net_sc3_weight = copy.deepcopy(net.sc3.weight.data)

		###serializing net 
	net_state_dict = net.state_dict()

		###Sparse Coding
		###make a copy for net, and then optimize sparsity level via copied net
	copy_net = copy.deepcopy(net)
	copy_state_dict = copy_net.state_dict()
	for name, param in copy_state_dict.items():
			###omit the param if it is not a weight matrix
		if not "weight" in name:
			continue
			###omit gene layer
		if "sc1" in name:
			continue
			###stop sparse coding
		if "sc4" in name:
			break
			###sparse coding between the current two consecutive layers is in the trained small sub-network
		if "sc2" in name:
			active_param = net_sc2_weight.mul(do_m1_grad_mask)
		if "sc3" in name:
			active_param = net_sc3_weight.mul(do_m2_grad_mask)
		nonzero_param_1d = active_param[active_param != 0]
		if nonzero_param_1d.size(0) == 0: ###stop sparse coding between the current two consecutive layers if there are no valid weights
			break
		copy_param_1d = copy.deepcopy(nonzero_param_1d)
			###set up potential sparsity level in [0, 100)
		S_set =  torch.arange(100, -1, -1)[1:]
		copy_param = copy.deepcopy(active_param)
		S_loss = []
		for S in S_set:
			param_mask = s_mask(sparse_level = S.item(), param_matrix = copy_param, nonzero_param_1D = copy_param_1d, dtype = dtype)
			transformed_param = copy_param.mul(param_mask)
			copy_state_dict[name].copy_(transformed_param)
			copy_net.train()
			y_tmp = copy_net(train_x, train_age)
			loss_tmp = neg_par_log_likelihood(y_tmp, train_ytime, train_yevent)
			S_loss.append(loss_tmp)
			###apply cubic interpolation
		interp_S_loss = interp1d(S_set, S_loss, kind='cubic')
		interp_S_set = torch.linspace(min(S_set), max(S_set), steps=100)
		interp_loss = interp_S_loss(interp_S_set)
		optimal_S = interp_S_set[np.argmin(interp_loss)]
		optimal_param_mask = s_mask(sparse_level = optimal_S.item(), param_matrix = copy_param, nonzero_param_1D = copy_param_1d, dtype = dtype)
		if "sc2" in name:
			final_optimal_param_mask = torch.where(do_m1_grad_mask == 0, torch.ones_like(do_m1_grad_mask), optimal_param_mask)
			optimal_transformed_param = net_sc2_weight.mul(final_optimal_param_mask)
		if "sc3" in name:
			final_optimal_param_mask = torch.where(do_m2_grad_mask == 0, torch.ones_like(do_m2_grad_mask), optimal_param_mask)
			optimal_transformed_param = net_sc3_weight.mul(final_optimal_param_mask)
			###update weights in copied net
		copy_state_dict[name].copy_(optimal_transformed_param)
			###update weights in net
		net_state_dict[name].copy_(optimal_transformed_param)

	if epoch % 200 == 0: 
		net.train()
		train_pred = net(train_x, train_age)
		train_loss = neg_par_log_likelihood(train_pred, train_ytime, train_yevent).view(1,)

		net.eval()
		eval_pred = net(eval_x, eval_age)
		eval_loss = neg_par_log_likelihood(eval_pred, eval_ytime, eval_yevent).view(1,)

		train_cindex = c_index(train_pred, train_ytime, train_yevent)
		eval_cindex = c_index(eval_pred, eval_ytime, eval_yevent)
		print("Loss in Train: ", train_loss)
