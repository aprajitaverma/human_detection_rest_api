from django.conf.urls import url
from . import views

urlpatterns = [
    url('human_detection/', views.DetectAPI.as_view()),
]
