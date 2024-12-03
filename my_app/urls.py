from django.urls import path

from my_app import views

urlpatterns = [
    path("", views.main, name="main"),
]