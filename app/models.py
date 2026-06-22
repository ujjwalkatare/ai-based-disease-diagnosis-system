from django.db import models

class Patient(models.Model):

    # 🔹 Basic Info
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    mobile = models.CharField(max_length=15)
    aadhaar = models.CharField(max_length=12, unique=True)

    # 🔹 Authentication
    password = models.CharField(max_length=128)

    # 🔹 Personal Health Info
    age = models.PositiveIntegerField(null=True, blank=True)
    gender = models.CharField(
        max_length=10,
        choices=[
            ('Male', 'Male'),
            ('Female', 'Female'),
            ('Other', 'Other')
        ],
        null=True,
        blank=True
    )

    # 🔹 System Fields (VERY IMPORTANT FOR PROJECT)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

# 🔥 NEW MODEL (HISTORY)
class PatientHistory(models.Model):

    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)

    disease_name = models.CharField(max_length=100)

    redirect_url = models.CharField(max_length=100)

    # optional (good for future)
    symptoms = models.TextField(null=True, blank=True)
    final_result = models.CharField(max_length=20, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.patient.name} - {self.disease_name}"