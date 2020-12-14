import face_recognition
import math


def ManhattanDistance(point1,point2):
	aux = 0
	for i in range(len(point1)):
		aux = aux + abs(point1[i]-point2[i])
	return aux

def EuclideanDistance(point1, point2):
	aux = 0
	for i in range(len(point1)):
		aux = aux + math.pow(point1[i]-point2[i],2)
	return math.sqrt(aux)


def face_Detect(image_path):
	img = face_recognition.load_image_file(image_path)
	vec = face_recognition.api.face_encodings(img)
	return vec

