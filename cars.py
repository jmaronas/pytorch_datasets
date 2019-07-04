import torch
import torch.utils.data as data
from scipy.misc import imresize,imsave
import scipy.io as sio
from PIL import Image
import os
import numpy
from progress.bar import Bar
import urllib.request
from progress.bar import Bar

from common import *

class cars_standford(ImageFolder):
	"""`Standford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html/>`_ Dataset.
	    Args:
        	root (string): Root directory of dataset where directory
	            ``cars_ims.tgz`` exists or will be saved to if download is set to True.
        	partition (string): Select partition: train/test.
		image_shape (int,optional): Number specifying the shape of the final image, default is 300
		interpolation(string,optional): Which interpolation is used when resize to image_shape is performed (default 'bilinear')
		padding(string,optional): which kind of padding is used, see numpy.pad (default wrap)
        	transform (callable, optional): A function/transform that takes in an PIL image
	            and returns a transformed version. E.g, ``transforms.RandomCrop``
	        target_transform (callable, optional): A function/transform that takes in the
        	    target and transforms it.
	        download (bool, optional): If true, downloads the dataset from the internet and
        	    puts it in root directory. If dataset is already downloaded, it is not
	            downloaded again.
	"""


	def __init__(self,directory,partition, image_shape=300,interpolation='bilinear',padding='wrap', download=True, transform=None, target_transform=None):

		if partition not in ['train','test']:
			raise ValueError("Unknown {} partition. Choose from train/test".format(partition))

		self.md5sum='d5c8f0aa497503f355e17dc7886c3f14'
		self.url='http://imagenet.stanford.edu/internal/car196/car_ims.tgz'	
		self.url_annot='http://imagenet.stanford.edu/internal/car196/cars_annos.mat'
		self.filename='car_ims'
		self.directory=directory
		self.processed_directory=os.path.join(self.directory,'cars_images_processed',str(image_shape),interpolation,padding)
		self.partition=partition
		self.image_shape=image_shape
		self.interpolation=interpolation
		self.padding=padding
		self.transform=transform
		self.target_transform=target_transform

		if download:
			self._download()

		#check everything is correctly downloaded
		if not self._check_integrity():
			raise Exception("Files corrupted. Set download=True") 

		if not self._check_if_process():
			self._process() 

		super(cars_standford, self).__init__(os.path.join(self.processed_directory,partition),transform=transform,target_transform=target_transform)


	def _check_integrity(self):
		fpath=os.path.join(self.directory,self.filename+'.tgz')
		if not check_integrity(fpath, self.md5sum):
                        return False
		return True

	def _download(self):

		if self._check_integrity():
			print('Files already downloaded and verified')
			return

		download_and_extract_archive(self.url, self.directory, filename=self.filename+'.tgz', md5=self.md5sum)
		urllib.request.urlretrieve(self.url_annot,os.path.join(self.directory,'cars_annos.mat'))  

	def _check_if_process(self):
		#first check if the path exits
		if not os.path.exists(self.processed_directory):
			return False
		
		#check if the process correctly finished
		path_file=os.path.join(self.processed_directory,'.correctly_processed')
		if not os.path.isfile(path_file):
			raise Exception("You seem to already have a folder named {} where processing did not succed. Erase folder and re-run".format(self.processed_directory))	

		return True

	def _process(self):

		print("Processing data. May take long the first time...")
		bar = Bar('Processing', max=8144+8041, suffix='Images %(index)d/%(max)d - %(percent).1f%% - %(eta)ds')

		train_dir=os.path.join(self.processed_directory,'train')
		test_dir=os.path.join(self.processed_directory,'test')
		file_bbox=os.path.join(self.directory,'cars_annos.mat')

		os.makedirs(train_dir)
		os.makedirs(test_dir)

		for i in range(196):		
			os.makedirs(os.path.join(train_dir,str(i)))
			os.makedirs(os.path.join(test_dir,str(i)))
	

		counter_train,counter_test=0,0

		for line in sio.loadmat(file_bbox)['annotations'][0]:
	
			image_name,a,b,c,d,label,is_test=line
			label=label[0,0]
			label=numpy.int64(label)-1
			
			x,y,width,height,is_test=int(a),int(b),int(c),int(d),int(is_test)

			image = numpy.array(Image.open(os.path.join(self.directory,image_name[0])))


			image = image[y:height,x:width]
			row=image.shape[0]
			col=image.shape[1]
			#padding
			if row>col:
				max_pad_col=row
				pad_size_col = (max_pad_col-col)/2.
				pad_size_col_l=numpy.ceil(pad_size_col).astype(numpy.int32)
				pad_size_col_r=numpy.floor(pad_size_col).astype(numpy.int32)
				pad_size_row_l,pad_size_row_r=0,0
			elif row<col:
				max_pad_row=col
				pad_size_row = (max_pad_row-row)/2.
				pad_size_row_l=numpy.ceil(pad_size_row).astype(numpy.int32)
				pad_size_row_r=numpy.floor(pad_size_row).astype(numpy.int32)
				pad_size_col_l,pad_size_col_r=0,0
			else:
				pad_size_row_l,pad_size_row_r,pad_size_col_l,pad_size_col_r=0,0,0,0

	
			if len(image.shape)==2:
				image = numpy.stack((image,)*3, -1)

			image=numpy.pad(image,((pad_size_row_l,pad_size_row_r),(pad_size_col_l,pad_size_col_r),(0,0)),self.padding)
			im_crop=imresize(image,(self.image_shape,self.image_shape,3),self.interpolation)


			if not is_test:
				imsave(os.path.join(train_dir,str(label),str(counter_train)+'.png'), im_crop)
				counter_train+=1
			else:
				imsave(os.path.join(test_dir,str(label),str(counter_test)+'.png'), im_crop)	
				counter_test+=1

		
			bar.next()

		bar.finish()
		path=os.path.join(self.processed_directory,'.correctly_processed')
		open(path,'w').close()

		print("Finish processing data")	 




class bbox_cars_standford(data.Dataset):
	"""`Standford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html/>`_ Dataset.
	    Args:
        	root (string): Root directory of dataset where directory
	            ``cars_ims.tgz`` exists or will be saved to if download is set to True.
        	partition (string): Select partition: train/test.
        	transform (callable, optional): A function/transform that takes in an PIL image
	            and returns a transformed version. E.g, ``transforms.RandomCrop``
	        target_transform (callable, optional): A function/transform that takes in the
        	    target and transforms it.
	        download (bool, optional): If true, downloads the dataset from the internet and
        	    puts it in root directory. If dataset is already downloaded, it is not
	            downloaded again.
	"""


	def __init__(self,directory,partition,download=True, transform=None, target_transform=None):
		super(bbox_cars_standford, self).__init__()

		if partition not in ['train','test']:
			raise ValueError("Unknown {} partition. Choose from train/test".format(partition))

		self.md5sum='d5c8f0aa497503f355e17dc7886c3f14'
		self.url='http://imagenet.stanford.edu/internal/car196/car_ims.tgz'	
		self.url_annot='http://imagenet.stanford.edu/internal/car196/cars_annos.mat'
		self.filename='car_ims'
		self.directory=directory
		self.partition=partition
		self.transform=transform
		self.target_transform=target_transform

		if download:
			self._download()

		#check everything is correctly downloaded
		if not self._check_integrity():
			raise Exception("Files corrupted. Set download=True") 

		self._process() 

	def default_collate(self,batch):
		data=[item[0] for item in batch]
		target=[item[1] for item in batch]
		bbox=[item[2] for item in batch]
		return data,target,bbox


	def _check_integrity(self):
		fpath=os.path.join(self.directory,self.filename+'.tgz')
		if not check_integrity(fpath, self.md5sum):
                        return False
		return True

	def _download(self):

		if self._check_integrity():
			print('Files already downloaded and verified')
			return

		download_and_extract_archive(self.url, self.directory, filename=self.filename+'.tgz', md5=self.md5sum)
		urllib.request.urlretrieve(self.url_annot,os.path.join(self.directory,'cars_annos.mat'))  

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):


		image_dir,target,bbox=self.images[index],self.labels[index],self.bbox[index]

		image=Image.open(image_dir)		

		if len(image.split())==2: #some images are grayscale
			image = Image.merge("RGB",(image,image,image))
			#image = numpy.stack((image,)*3, -1)

		if self.transform is not None:
			sample = self.transform(image)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return sample,target,torch.from_numpy(bbox)



	def _process(self):
		print("Processing data...")
		bar = Bar('Processing', max=8144+8041, suffix='Images %(index)d/%(max)d - %(percent).1f%% - %(eta)ds')


		file_bbox=os.path.join(self.directory,'cars_annos.mat')
		train_images_path,train_labels,test_images_path,test_labels=[],[],[],[]	
		train_bbox = -1*numpy.ones((8144,4),dtype=numpy.int32)
		test_bbox = -1*numpy.ones((8041,4),dtype=numpy.int32)

		counter_train,counter_test=0,0

		
		for line in sio.loadmat(file_bbox)['annotations'][0]:
	
			image_name,a,b,c,d,label,is_test=line

			label=label[0,0]
			label=numpy.int64(label)-1
			
			x,y,width,height,is_test=int(a),int(b),int(c),int(d),int(is_test)

			image_dir=os.path.join(self.directory,image_name[0])
		
			if not is_test:
				train_bbox[counter_train,:]=[x,y,width,height]
				train_images_path.append(image_dir)
				train_labels.append(label)

				counter_train+=1
			else:
				test_bbox[counter_test,:]=[x,y,width,height]
				test_images_path.append(image_dir)
				test_labels.append(label)

				counter_test+=1
		
			bar.next()

		bar.finish()
		print("Finish processing data")	


		if self.partition=='train':
			self.images=train_images_path
			self.labels=train_labels
			self.bbox=train_bbox

		else:
			self.images=test_images_path
			self.labels=test_labels
			self.bbox=test_bbox






	
