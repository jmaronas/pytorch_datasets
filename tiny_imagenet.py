import torch
import torch.utils.data as data

from torchvision.datasets.utils import * #check_integrity , download_and_extract_archive, extract_archive and so on
from torchvision.datasets.folder import ImageFolder

from scipy.misc import imresize,imsave
from PIL import Image
import os
import zipfile


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)

def download_and_extract_archive(url, download_root, extract_root=None, filename=None,
                                 md5=None, remove_finished=False):
	download_root = os.path.expanduser(download_root)
	if extract_root is None:
		 extract_root = download_root
	if not filename:
		filename = os.path.basename(url)

	#download_url(url, download_root, filename, md5)

	archive = os.path.join(download_root, filename)

	print("Extracting {} to {}".format(archive, extract_root))
	extract_archive(archive, extract_root, remove_finished)



#TODO: return bounding boxes, allow for other types of interpolations
class tiny_ImageNet(ImageFolder):
	"""`tiny ImageNet <https://tiny-imagenet.herokuapp.com/>`_ Dataset.
	    Args:
        	root (string): Root directory of dataset where directory
	            ``tiny-imagenet-200.zip`` exists or will be saved to if download is set to True.
        	partition (string): Select partition: train/valid/test. If test is selected then 
	            a unique label 0 is returned.
		image_shape (int,optional): Number specifying the shape of the final image, default is 64.
        	transform (callable, optional): A function/transform that takes in an PIL image
	            and returns a transformed version. E.g, ``transforms.RandomCrop``
	        target_transform (callable, optional): A function/transform that takes in the
        	    target and transforms it.
	        download (bool, optional): If true, downloads the dataset from the internet and
        	    puts it in root directory. If dataset is already downloaded, it is not
	            downloaded again.
	"""


	def __init__(self,directory, partition, image_shape=64,transform=None, target_transform=None,download=True):


		self.md5sum='90528d7ca1a48142e341f4ef8d21d0de'
		self.url='http://cs231n.stanford.edu/tiny-imagenet-200.zip'
		self.directory=directory
		self.filename='tiny-imagenet-200'
		self.partition=partition
		self.image_shape=image_shape


		self.interpolation='bilinear'
		self.transform=transform
		self.target_transform=target_transform



		if download:
			self._download()

		#check everything is correctly downloaded
		if not self._check_integrity():
			raise Exception("Files corrupted. Set download=True")			

		#one can decide to reshape images after download, instead of doing online, which will increse computation
		if not self._check_if_process():
			self._process()	

		split_folder=os.path.join(self.directory,self.filename,'processed',str(self.image_shape),self.partition,'images')
		super(tiny_ImageNet, self).__init__(split_folder,transform=transform,target_transform=target_transform)

	
	def _get_classes(self):
		path=os.path.join(self.directory,self.filename,'wnids.txt')
		return [line.split('\n')[0] for line in open(path,'r')]
			
	def _create_directories(self,class_list):
		for _ in class_list:
			path=os.path.join(self.directory,self.filename,'processed',str(self.image_shape),self.partition,'images',_)
			os.makedirs(path)

	def _process_train(self,class_list):
		print("Processing Train Images")
		for c in class_list:
			path=os.path.join(self.directory,self.filename,'train',c,'images')

			for f in os.listdir(path):
				fpath=os.path.join(self.directory,self.filename,'train',c,'images',f)
				image = Image.open(fpath)

				if self.image_shape!=64:
					new_im=imresize(image,(self.image_shape,self.image_shape),self.interpolation)
				else:
					new_im=image

				save_path=os.path.join(self.directory,self.filename,'processed',str(self.image_shape),self.partition,'images',c,f)
				imsave(save_path+'.JPEG',new_im)



	def _process_valid(self):

		print("Processing Validation Images")
		val_file = os.path.join(self.directory,self.filename,'val','val_annotations.txt')
		image_dir = os.path.join(self.directory,self.filename,'val','images') 

		for line in open(val_file,'r'):
			image_name,image_class,_,_,_,_=line.split()

			fpath=os.path.join(image_dir,image_name)

			image = Image.open(fpath)

			if self.image_shape!=64:
				new_im=imresize(image,(self.image_shape,self.image_shape),self.interpolation)
			else:
				new_im=image

			save_path=os.path.join(self.directory,self.filename,'processed',str(self.image_shape),self.partition,'images',image_class,image_name)
			imsave(save_path+'.JPEG',new_im)



	def _process_test(self):
		path=os.path.join(self.directory,self.filename,'test','images')
		print("Processing Test Images")
		for f in os.listdir(path):
			fpath=os.path.join(self.directory,self.filename,'test','images',f)
			image = Image.open(fpath)
			if self.image_shape!=64:
				new_im=imresize(image,(self.image_shape,self.image_shape),self.interpolation)
			else:
				new_im=image


			save_path=os.path.join(self.directory,self.filename,'processed',str(self.image_shape),self.partition,'images','0',f)
			imsave(save_path+'.JPEG',new_im)



	def _process(self):
		path=os.path.join(self.directory,self.filename,'processed',str(self.image_shape),self.partition,'images')
		os.makedirs(path)		
		class_lists=self._get_classes()

		if self.partition in ['train','valid']:
			self._create_directories(class_lists)
		else: 
			path=os.path.join(self.directory,self.filename,'processed',str(self.image_shape),self.partition,'images','0')
			os.makedirs(path)

		if self.partition=='train':
			self._process_train(class_lists)

		elif self.partition=='valid':
			self._process_valid()
			
		elif self.partition=='test':
			self._process_test()

		else:
			raise ValueError('Invalid {} partition name. Choose from train valid or test'.format(partition))
		
		path=os.path.join(self.directory,self.filename,'processed',str(self.image_shape),self.partition,'.correctly_processed')
		open(path,'w').close()

	def _check_if_process(self):
		#first check if the path exits
		path=os.path.join(self.directory,self.filename,'processed',str(self.image_shape),self.partition)

		if not os.path.exists(path):
			return False
		
		#check if the process correctly finished
		path_file=os.path.join(self.directory,self.filename,'processed',str(self.image_shape),self.partition,'.correctly_processed')
		if not os.path.isfile(path_file):
			raise Exception("You seem to already have a folder named {} where processing did not succed. Erase folder and re-run".format(path))	

		return True

	def _check_integrity(self):
		fpath=os.path.join(self.directory,self.filename+'.zip')
		if not check_integrity(fpath, self.md5sum):
                        return False
		return True

	def _download(self):
		'''
		if self._check_integrity():
			print('Files already downloaded and verified')
			return
		'''
		download_and_extract_archive(self.url, self.directory, filename=self.filename+'.zip', md5=self.md5sum)




