import torch


def Data_To_Dataloader(trainset,testset,seed=1,batch_size=128, num_workers=2,pin_memory=True):
	"""
	Converts DataSet Object to DataLoader
	"""
	SEED = 1
	cuda = torch.cuda.is_available()
	torch.manual_seed(SEED)

	if cuda:
			torch.cuda.manual_seed(SEED)

	dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=64)
	trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
	testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
	return  trainloader, testloader
