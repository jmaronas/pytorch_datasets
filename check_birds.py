import torch
import torchvision.transforms as tvtr

import matplotlib.pyplot as plt
import time

from birds import birds_caltech_2011,bbox_birds_caltech_2011

dataset = birds_caltech_2011('./data/','train',image_shape=200,download=True,transform=tvtr.ToTensor())
data_train =  torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)

for idx,(x,t) in enumerate(data_train):
	x=x[0].numpy().transpose((1,2,0))
	t=t[0]

	plt.imshow(x)
	plt.title('Class {}'.format(t.item()))
	plt.pause(2)

	if idx==10:
		break

plt.close()

dataset = bbox_birds_caltech_2011('./data/','train',download=True,transform=tvtr.ToTensor())
data_train =  torch.utils.data.DataLoader(dataset,batch_size=1,collate_fn=dataset.default_collate,shuffle=True)

fig=plt.figure(figsize=(1, 2))
for idx,(x,t,bbox) in enumerate(data_train):
	image=x[0].numpy().transpose((1,2,0))
	t=t[0]
	x,y,width,height=bbox[0]
	x,y,width,height=x.item(),y.item(),width.item(),height.item()

	fig.add_subplot(1, 2, 1)
	plt.imshow(image)
	plt.title('Class {}'.format(t.item()))

	fig.add_subplot(1, 2, 2)
	plt.imshow(image[y:y+height,x:x+width])
	plt.title('Class {}'.format(t.item()))

	plt.pause(2)

	if idx==10:
		break



