from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.template import Context, loader
from django.views.decorators.csrf import csrf_exempt

import datetime

from src.sequential_knn import SequentialKNN

AUX_MEDIA_DIR = 'store/aux/'

sequential_knn = SequentialKNN('store/models/points.pkl')

@csrf_exempt
def index(request):
	template = loader.get_template("index.html")
	return HttpResponse(template.render())

@csrf_exempt
@require_http_methods(["POST"])
def search(request):
	image = request.FILES.get('file')
	k = int(request.POST.get('k'))

	file_path = AUX_MEDIA_DIR + str(datetime.datetime.now().time()) + '.png'
	filename = default_storage.save(file_path, image)

	nns = sequential_knn.KNN_Faces_ED(filename, k)

	default_storage.delete(file_path)

	freq = {}
	for face in nns:
		if face not in freq:
			freq[face] = 0
		freq[face] += 1

	return JsonResponse({'nns' : freq})

