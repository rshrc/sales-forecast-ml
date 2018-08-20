from django.db import models


# Create your models here.

class Product(models.Model):
    back_camera = models.DecimalField(decimal_places=2, max_digits=10)
    front_camera = models.DecimalField(decimal_places=2, max_digits=10)
    resolution_1 = models.DecimalField(decimal_places=2, max_digits=10)
    resolution_2 = models.DecimalField(decimal_places=2, max_digits=10)
    screen_size = models.DecimalField(decimal_places=2, max_digits=10)
    battery = models.DecimalField(decimal_places=2, max_digits=10)
    price = models.DecimalField(decimal_places=2, max_digits=10)
    pre_release_demand = models.DecimalField(decimal_places=2, max_digits=10)
    # sales = models.DecimalField(decimal_places=2, max_digits=10)
    # quarter = models.DecimalField(decimal_places=2, max_digits=10)
