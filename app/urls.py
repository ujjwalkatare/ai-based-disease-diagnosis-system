from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),

    path("register/", views.register, name="register"),
    path("login/", views.login_view, name="login"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("profile/", views.profile, name="profile"),
    path("patient-check/", views.patient_check, name="patient_check"),
    path('save-report/', views.save_report, name='save_report'),
    path("history/", views.history, name="history"),
    path('check-test/', views.check_test, name='check_test'),
    path('redirect-test/<str:test_name>/', views.redirect_test, name='redirect_test'),
    path('analytics/', views.analytics, name='analytics'),
    path("logout/", views.logout_view, name="logout"),

    path('diabetes/', views.diabetes_predict, name='diabetes_predict'),
    path('heart/', views.heart_predict, name='heart_predict'),
    path('kidney/', views.kidney_predict, name='kidney_predict'),
    path('liver/', views.liver_predict, name='liver_predict'),
    path('malaria/', views.malaria_predict, name='malaria'),
    path('pneumonia/', views.pneumonia_predict, name='pneumonia'),
]