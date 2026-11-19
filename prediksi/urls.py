
from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("api/prediksi/", views.api_prediksi, name="api_prediksi"),
    path("api/grafik/", views.api_grafik, name="api_grafik"),
    path("api/test-groq/", views.test_groq, name="test_groq"),
]
