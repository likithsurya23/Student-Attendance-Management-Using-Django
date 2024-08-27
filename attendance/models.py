from django.db import models

class Student(models.Model):
    # Define fields for the Student model
    id = models.AutoField(primary_key=True)  # Auto-generated primary key
    name = models.CharField(max_length=100)  # Name of the student
    date_of_birth = models.DateField()  # Date of birth of the student
    email = models.EmailField(unique=True, blank=True, null=True)  # Email address of the student

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Student"
        verbose_name_plural = "Students"
        ordering = ['name']
