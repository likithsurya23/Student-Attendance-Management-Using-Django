from django.urls import path
from .views import login_view
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('start/', views.start, name='start'),
    path('add/', views.add_user, name='add_user'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('edit/<int:student_id>/', views.edit_student, name='edit_student'),
    path('delete/<int:student_id>/', views.delete_student, name='delete_student'),
    path('login/', views.login_view, name='login'),

]
