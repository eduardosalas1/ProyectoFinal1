import face_recognition
import math
import pickle

from src.functions import *
 
class SequentialKNN:
	def __init__(self, result_file):
		with open(result_file,'rb') as f:
			self.faces = pickle.load(f)
			self.faces_partition = self.faces

	def resize_partition(self,n):
		self.faces_partition = self.faces[:n]

	def KNN_Faces_ED(self, image_path, n):
		vec = face_Detect(image_path)
		if len(vec) == 0:
			print("The image couldn't be identified")
		else:
			ed_order = sorted(self.faces_partition,key=lambda x: EuclideanDistance(vec[0],x[1]))
		return [item[0] for item in ed_order[:n]]

	def KNN_Faces_MD(self,image_path,n):
		vec = face_Detect(image_path)
		if len(vec) == 0:
			print("The image couldn't be identified")
		else:
			md_order = sorted(self.faces_partition,key=lambda x: ManhattanDistance(vec[0],x[1]))
		return [item[0] for item in md_order[:n]]

