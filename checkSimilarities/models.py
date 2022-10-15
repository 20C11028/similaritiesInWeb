from django.db import models

# Create your models here.
class Pair(models.Model):
    name = models.CharField(max_length=100)
    hotel_Main_Img = models.ImageField(upload_to='images/')