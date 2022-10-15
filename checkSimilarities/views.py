from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from django.core.files.storage import FileSystemStorage

import torch
import clip
from PIL import Image
import numpy as np
import keras.backend as kb

context = {
    'name': 'Nobody',
  }

def index(request):
  template = loader.get_template('myfirst.html')
  return HttpResponse(template.render(context, request))

def upload(request):
  print("Upload!!")
  if request.method == 'POST' and request.FILES['image']:
    text = request.POST['text']
    print(text)
    upload = request.FILES['image']
    fss = FileSystemStorage()
    file = fss.save(upload.name, upload)
    file_url = fss.url(file)
    file_url = file_url.replace("/media", "media")
    cos, euc, man = calScore(text, file_url, "ViT-B/32")
    global context
    context = {'text': text, 'file_url': file_url, 'cos': cos, 'euc': euc, 'man': man}
    return HttpResponseRedirect(reverse('index'))
  return HttpResponseRedirect(reverse('index'))

def extractFeature(text, imageUrl, model):
  model, preprocess = clip.load(model, device='cuda')
  image = Image.open(imageUrl)
  image_input = preprocess(image).unsqueeze(0).cuda()
  text_input = clip.tokenize(text).cuda()
  with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_input).float()

  return text_features, image_features

def calScore(text, imageUrl, model):
  text_features, image_features = extractFeature(text, imageUrl, model)
  cosForm = torch.nn.CosineSimilarity(dim=1)
  cos = cosForm(text_features, image_features)
  text_features = text_features.cpu()
  image_features = image_features.cpu()
  euc = np.linalg.norm(text_features - image_features)
  man = kb.sum( kb.abs(text_features - image_features),axis=1,keepdims=True)
  print(cos, euc, man)
  return cos[0], euc, kb.get_value(man)