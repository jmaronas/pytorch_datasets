import torch
import torch.utils.data as data
from scipy.misc import imresize,imsave
from PIL import Image
import os
import numpy
from progress.bar import Bar

from common import *

class birds_caltech_2011(ImageFolder):
	"""`Caltech 2011 Birds <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html/>`_ Dataset.
	    Args:
        	root (string): Root directory of dataset where directory
	            ``CUB_200_2011.tgz`` exists or will be saved to if download is set to True.
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

		self.md5sum='97eceeb196236b17998738112f37df78'
		self.url='http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'	
		self.filename='CUB_200_2011'
		self.directory=directory
		self.processed_directory=os.path.join(self.directory,self.filename,'processed',str(image_shape),interpolation,padding)
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

		super(birds_caltech_2011, self).__init__(os.path.join(self.processed_directory,partition),transform=transform,target_transform=target_transform)
		

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
		bar = Bar('Processing', max=5994+5794, suffix='Images %(index)d/%(max)d - %(percent).1f%% - %(eta)ds')

		file_images=os.path.join(self.directory,self.filename,'images.txt')
		file_train_test=os.path.join(self.directory,self.filename,'train_test_split.txt')
		file_classes=os.path.join(self.directory,self.filename,'image_class_labels.txt')
		file_bbox=os.path.join(self.directory,self.filename,'bounding_boxes.txt')
		image_directory=os.path.join(self.directory,self.filename,'images')

		train_dir=os.path.join(self.processed_directory,'train')
		test_dir =os.path.join(self.processed_directory,'test')
		os.makedirs(train_dir)
		os.makedirs(test_dir)

		for i in range(200):		
			os.makedirs(os.path.join(train_dir,str(i)))
			os.makedirs(os.path.join(test_dir,str(i)))
	

		counter_train,counter_test=0,0
		for l1,l2,l3,l4 in zip(open(file_images),open(file_train_test),open(file_classes),open(file_bbox)):
			image_name=l1.split("\n")[0].split(" ")[1]
			is_train = int(l2.split("\n")[0].split(" ")[1])
			label    = numpy.int64(l3.split("\n")[0].split(" ")[1])-1

			x,y,width,height= l4.split("\n")[0].split(" ")[1:]
			x,y,width,height=numpy.float32(x).astype('int32'),numpy.float32(y).astype('int32'),numpy.float32(width).astype('int32'),numpy.float32(height).astype('int32')


			image_dir= os.path.join(image_directory,image_name)
			image = numpy.array(Image.open(image_dir))
			
			image=image[y:y+height,x:x+width]
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

			if len(image.shape)==2: #some images are grayscale
				image = numpy.stack((image,)*3, -1)

			image=numpy.pad(image,((pad_size_row_l,pad_size_row_r),(pad_size_col_l,pad_size_col_r),(0,0)),self.padding)
			im_crop=imresize(image,(self.image_shape,self.image_shape,3),self.interpolation)	
				
			if is_train:
				save_to=os.path.join(train_dir,str(label),str(counter_train)+'.png')
				imsave(save_to, im_crop)
				counter_train+=1
			else:
				save_to=os.path.join(test_dir,str(label),str(counter_test)+'.png')
				imsave(save_to, im_crop)	
				counter_test+=1

			bar.next()


		bar.finish()
		path=os.path.join(self.processed_directory,'.correctly_processed')
		open(path,'w').close()
		print("Finish processing data")	




class bbox_birds_caltech_2011(data.Dataset):
	"""`Caltech 2011 Birds <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html/>`_ Dataset.
	    Args:
        	root (string): Root directory of dataset where directory
	            ``CUB_200_2011.tar.gz`` exists or will be saved to if download is set to True.
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
		super(bbox_birds_caltech_2011, self).__init__()

		if partition not in ['train','test']:
			raise ValueError("Unknown {} partition. Choose from train/test".format(partition))

		self.md5sum='97eceeb196236b17998738112f37df78'
		self.url='http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'	
		self.filename='CUB_200_2011'
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

	def default_collate(self,batch):
		data=[item[0] for item in batch]
		target=[item[1] for item in batch]
		bbox=[item[2] for item in batch]
		return data,target,bbox

	def _prepare_for_loading(self):
		name='birds_train_bbox.npz' if self.partition=='train' else 'birds_test_bbox.npz'
		bounding_box = numpy.load(os.path.join(self.processed_directory,name))['bbox']
		root=os.path.join(self.processed_directory,self.partition)
		classes, class_to_idx = self._find_classes(root)
		samples = make_dataset_withbbox(root, class_to_idx, IMG_EXTENSIONS,bounding_box)
		self.samples = samples
		self.targets = [s[1] for s in samples]				

	def _process(self):
		print("Processing data. May take long the first time...")
		bar = Bar('Processing', max=5994+5794, suffix='Images %(index)d/%(max)d - %(percent).1f%% - %(eta)ds')

		file_images=os.path.join(self.directory,self.filename,'images.txt')
		file_train_test=os.path.join(self.directory,self.filename,'train_test_split.txt')
		file_classes=os.path.join(self.directory,self.filename,'image_class_labels.txt')
		file_bbox=os.path.join(self.directory,self.filename,'bounding_boxes.txt')
		image_directory=os.path.join(self.directory,self.filename,'images')

		train_images_path,train_labels,test_images_path,test_labels=[],[],[],[]	
		train_bbox = -1*numpy.ones((5994,4),dtype=numpy.int32)
		test_bbox = -1*numpy.ones((5794,4),dtype=numpy.int32)

		counter_train,counter_test=0,0
		
		for l1,l2,l3,l4 in zip(open(file_images),open(file_train_test),open(file_classes),open(file_bbox)):
			image_name=l1.split("\n")[0].split(" ")[1]
			is_train = int(l2.split("\n")[0].split(" ")[1])
			label    = numpy.int64(l3.split("\n")[0].split(" ")[1])-1

			x,y,width,height= l4.split("\n")[0].split(" ")[1:]
			x,y,width,height=numpy.float32(x).astype('int32'),numpy.float32(y).astype('int32'),numpy.float32(width).astype('int32'),numpy.float32(height).astype('int32')

			image_dir= os.path.join(image_directory,image_name)

			if is_train:
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

