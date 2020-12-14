import face_recognition
import math
import pickle
import time
 
def ED(vector1, vector2):
	res = 0
	for i in range(len(vector1)):
		res+= math.pow(vector1[i]-vector2[i],2)
	return math.sqrt(res)

def MD(vector1,vector2):
	res = 0
	for i in range(len(vector1)):
		res+= abs(vector1[i]-vector2[i])
	return res

def read_Face(image_path):
	img = face_recognition.load_image_file(image_path)
	vec = face_recognition.api.face_encodings(img)
	return vec

class Secuential_KNN:
	def __init__(self, result_file):
		with open(result_file,'rb') as f:
			self.faces = pickle.load(f)
			self.faces_partition = self.faces


	def resize_partition(self,n):
		self.faces_partition = self.faces[:n]

	def KNN_Faces_ED(self,image_path,n):
		vec = read_Face(image_path)
		if len(vec) == 0:
			print("The image couldn't be identified")
		else:
			ed_order = sorted(self.faces_partition,key=lambda x: ED(vec[0],x[1]))
		return [item[0] for item in ed_order[:n]]

	def KNN_Faces_MD(self,image_path,n):
		vec = read_Face(image_path)
		if len(vec) == 0:
			print("The image couldn't be identified")
		else:
			md_order = sorted(self.faces_partition,key=lambda x: MD(vec[0],x[1]))
		return [item[0] for item in md_order[:n]]

# Efficency
####################################
sknn = Secuential_KNN('result.pkl')

# print("ED:")
# print(sknn.KNN_Faces_ED('./unknownFaces/jennifer_lopez.jpg',16))
# print("--------------------------------------------")
# print("MD:")
# print(sknn.KNN_Faces_MD('./unknownFaces/jennifer_lopez.jpg',16))
###################################

#Times
###################################
sizes = [100,200,400,800,1600,3200,6400,12800]

for s in sizes:
	sknn.resize_partition(s)
	start_time = time.time()
	sknn.KNN_Faces_ED('./unknownFaces/nicole_kidman.jpg',19)
	print("--- %s seconds ---" % (time.time() - start_time))
