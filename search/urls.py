from django.urls import path
from . import views

app_name = "search"

urlpatterns = [
    path('', views.index, name="index"),
    path('content/<str:doc_id>/', views.content, name="content")
]