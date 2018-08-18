from django.urls import path, include

from . import views

urlpatterns = [
    path('', views.product_describe_view, name='product_add'),    
]
