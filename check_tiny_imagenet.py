import torch
import torchvision.transforms as tvtr

import matplotlib.pyplot as plt
from tiny_imagenet import tiny_ImageNet


dataset = tiny_ImageNet('./data/','train',image_shape=128,download=True,transform=tvtr.ToTensor())
data_train =  torch.utils.data.DataLoader(dataset,batch_size=1)


for x,t in data_train:
	x=x[0].numpy().transpose((1,2,0))

	plt.imshow(x)
	plt.show()

