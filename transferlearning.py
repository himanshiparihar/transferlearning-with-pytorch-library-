import os
import torch.nn as nn
import torch as T 
import torch.optim as optim
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import cv2

def setparm(model , extracting):
	if extracting:
		for par in model.parameters():
			par.require_grad = False

# def showImg(img):
# 	img = img.numpy()
# 	cv2.imshow("Image", img)
# 	cv2.waitKey(1000)


def trainmod(model ,device,data_loaders , criterion ,optimizer, epochs = 25):
	for epoch in range(epochs):
		print("epoch %d/%d" % (epoch,epochs-1))
		print("-"*15)

		for phase in ['train','val']:
			if phase == 'train':
				model.train()
			else:
				model.eval()
			running_loss = 0.0
			correct = 0

			for ip , lb in data_loaders[phase]:
				# cv2.imshow("Image" ,ip)
				# cv2.waitKey(1000)
				# cv2.imshow("Lable", lb)
				# cv2.waitKey(1000)
				ip= ip.to(device)
				lb = lb.to(device)
				# showImg(ip)
				# print(ip.type())
				optimizer.zero_grad()
				# only want to calculate gradient when in train mode
				with T.set_grad_enabled(phase=='train'): 
					outputs = model(ip)
					loss = criterion(outputs,lb)

					_, preds = T.max(outputs,1)

					if phase == 'train':
						loss.backward()
						optimizer.step()
				running_loss += loss.item() * ip.size(0)
				correct += T.sum(preds == lb.data )
			epoch_loss = running_loss / len(data_loaders[phase].dataset)
			epoch_acc = correct.double() / len(data_loaders[phase].dataset)

			print('phase , loss , Acc' , phase , epoch_loss , epoch_acc)

if __name__ == '__main__':
	root_dir = 'hymenoptera_data1/'
	image_transform = {
	'train': transforms.Compose([transforms.RandomRotation((-270,270)),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]),
	'val':  transforms.Compose([transforms.RandomRotation((-270,270)),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])}
	# it is a generator object that maps the transform to the image we load from directory 
	data_generator = {k : datasets.ImageFolder(os.path.join(root_dir,k),image_transform[k]) for k in ['train','val']}

	# every data generator has a data loader
	data_loader = {k : T.utils.data.DataLoader(data_generator[k],batch_size=2,shuffle = True , num_workers= 4) for k in ['train','val']}

	device = T.device('cuda')
	model = models.resnet18(pretrained = True)

	setparm(model,True)
	num_features = model.fc.in_features
	model.fc = nn.Linear(num_features , 2)
	model.to(device)

	criterion= nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters() , lr = .001)

	params_to_update = []
	for name , param in model.named_parameters():
		if param.requires_grad is True:
			params_to_update.append(param)
			print('/t',name)

	trainmod(model, device, data_loader , criterion , optimizer)