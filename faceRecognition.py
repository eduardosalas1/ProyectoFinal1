import face_recognition
import math
import numpy as np
import sys
import os
import pickle

PATH = "./lfw/"

faces = []
for subdir, dirs, files in os.walk(PATH):
	for direct in dirs:
		print(direct)
		for i in os.listdir(PATH + direct):
			img = face_recognition.load_image_file(PATH + direct + "/" + i)
			vec = face_recognition.api.face_encodings(img)
			if len(vec) > 0:
				tmp = [direct]
				tmp.append(vec[0].tolist())
				faces.append(tmp)


with open('result.pkl','wb') as f:
	pickle.dump(faces,f)

