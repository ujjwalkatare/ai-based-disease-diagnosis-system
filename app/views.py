import os
import re
import json
import pickle
import joblib
from datetime import timedelta
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import torch
import torch.nn as nn
from torchvision import models, transforms
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.core.files.storage import FileSystemStorage
from django.db.models import Q
from django.http import JsonResponse
from django.utils.timezone import now
from .models import Patient, PatientHistory
from .utils.gemini_helper import get_gemini_response

DIABETES_MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'app', 'ml_models', 'diabetes.pkl'
)
HEART_MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'app', 'ml_models', 'heart_disease_clean_model.joblib'
)
KIDNEY_MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'app', 'ml_models', 'kidney_disease_model.joblib'
)
LIVER_MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'app', 'ml_models', 'liver_disease_model.joblib'
)
MALARIA_MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'app', 'ml_models', 'malaria_model.h5'
)
PNEUMONIA_MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'app', 'ml_models', 'best_resnet50.pth'
)

diabetes_model = pickle.load(open(DIABETES_MODEL_PATH, 'rb'))
heart_model = joblib.load(HEART_MODEL_PATH)
kidney_model = joblib.load(KIDNEY_MODEL_PATH)
liver_model = joblib.load(LIVER_MODEL_PATH)
malaria_model = None


device = torch.device("cpu")

pneumonia_model = models.resnet50(pretrained=False)

num_ftrs = pneumonia_model.fc.in_features
pneumonia_model.fc = nn.Linear(num_ftrs, 2)

pneumonia_model.load_state_dict(torch.load(PNEUMONIA_MODEL_PATH, map_location=device))

pneumonia_model.to(device)
pneumonia_model.eval()

pneumonia_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# 🔹 REGISTER VIEW
def register(request):

    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        mobile = request.POST.get("mobile")
        aadhaar = request.POST.get("aadhaar")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")
        age = request.POST.get("age")
        gender = request.POST.get("gender")

        # =========================
        # 🔹 VALIDATIONS
        # =========================

        # ✅ Name check
        if not name or len(name) < 3:
            messages.error(request, "Name must be at least 3 characters")
            return redirect("register")

        # ✅ Email validation
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_pattern, email):
            messages.error(request, "Invalid email format")
            return redirect("register")

        # ✅ Mobile validation (10 digits only)
        if not mobile.isdigit() or len(mobile) != 10:
            messages.error(request, "Mobile number must be exactly 10 digits")
            return redirect("register")

        # ✅ Aadhaar validation (12 digits only)
        if not aadhaar.isdigit() or len(aadhaar) != 12:
            messages.error(request, "Aadhaar must be exactly 12 digits")
            return redirect("register")

        # ✅ Password validation
        password_pattern = r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{6,}$'
        if not re.match(password_pattern, password):
            messages.error(
                request,
                "Password must contain 1 uppercase, 1 number, 1 special character"
            )
            return redirect("register")

        # ✅ Confirm password
        if password != confirm_password:
            messages.error(request, "Passwords do not match!")
            return redirect("register")

        # =========================
        # 🔹 DUPLICATE CHECK
        # =========================

        if Patient.objects.filter(email=email).exists():
            messages.error(request, "Email already registered!")
            return redirect("register")

        if Patient.objects.filter(aadhaar=aadhaar).exists():
            messages.error(request, "Aadhaar already registered!")
            return redirect("register")

        if Patient.objects.filter(mobile=mobile).exists():
            messages.error(request, "Mobile already registered!")
            return redirect("register")

        # =========================
        # 🔹 SAVE DATA
        # =========================

        Patient.objects.create(
            name=name,
            email=email,
            mobile=mobile,
            aadhaar=aadhaar,
            password=password,
            age=age if age else None,
            gender=gender if gender else None
        )

        messages.success(request, "Registration Successful! Please Login.")
        return redirect("login")

    return render(request, "registration.html", {'show_header': True})

# 🔹 LOGIN VIEW

def login_view(request):

    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        try:
            user = Patient.objects.get(email=email)

            if user.password == password:
                # ✅ Store session
                request.session['patient_id'] = user.id
                request.session['patient_name'] = user.name

                messages.success(request, "Login Successful!")
                return redirect("dashboard")
            else:
                messages.error(request, "Invalid Password!")

        except Patient.DoesNotExist:
            messages.error(request, "User not found!")

        return redirect("login")

    return render(request, "login.html", {'show_header': True})


# 🔹 DASHBOARD VIEW
def dashboard(request):

    # 🔒 Login check
    if not request.session.get('patient_id'):
        return redirect("login")

    patient_id = request.session.get('patient_id')
    patient_name = request.session.get('patient_name')

    # 🔹 Get all records (latest first)
    records = PatientHistory.objects.filter(
        patient_id=patient_id
    ).order_by('-created_at')

    # =========================
    # 🔹 1. STATS CARDS
    # =========================
    total_tests = records.count()
    completed_tests = records.filter(final_result__isnull=False).count()
    pending_tests = records.filter(final_result__isnull=True).count()

    # =========================
    # 🔹 2. RECENT ACTIVITY
    # =========================
    recent_activities = records[:5]

    for r in recent_activities:
        r.result = r.final_result if r.final_result else "Pending"
        r.status = "Completed" if r.final_result else "Pending"
        r.date = r.created_at

    # =========================
    # 🔹 3. LAST RESULT
    # =========================
    last_result = records.filter(final_result__isnull=False).first()

    if last_result:
        last_result.result = last_result.final_result
        last_result.result_class = "success" if last_result.final_result == "Negative" else "danger"
        last_result.confidence = 90  # 🔥 You can replace with real model confidence
        last_result.date = last_result.created_at

    # =========================
    # 🔹 4. ALERTS
    # =========================
    alerts = []

    if pending_tests > 0:
        alerts.append({
            "type": "warning",
            "title": "Pending Tests",
            "message": "You have pending tests. Complete them soon!",
            "action_url": "/history/"
        })

    if completed_tests > 5:
        alerts.append({
            "type": "info",
            "title": "Good Progress",
            "message": "You are actively monitoring your health. Keep it up!",
            "action_url": None
        })

    # =========================
    # 🔹 FINAL RENDER
    # =========================
    return render(request, "dashboard.html", {
        "name": patient_name,
        "total_tests": total_tests,
        "completed_tests": completed_tests,
        "pending_tests": pending_tests,
        "recent_activities": recent_activities,
        "last_result": last_result,
        "alerts": alerts,
        "show_header": False
    })

def profile(request):

    # 🔒 Login check
    if not request.session.get('patient_id'):
        return redirect("login")

    patient_id = request.session.get('patient_id')

    # 🔹 Get patient
    patient = Patient.objects.get(id=patient_id)

    # 🔹 Get history records
    records = PatientHistory.objects.filter(
        patient_id=patient_id
    ).order_by('-created_at')

    # =========================
    # 🔹 STATS
    # =========================
    total_tests = records.count()
    completed_tests = records.filter(final_result__isnull=False).count()
    pending_tests = records.filter(final_result__isnull=True).count()

    # =========================
    # 🔹 RECENT ACTIVITIES
    # =========================
    recent_activities = records[:5]

    for r in recent_activities:
        r.result = r.final_result if r.final_result else "Pending"
        r.status = "Completed" if r.final_result else "Pending"
        r.date = r.created_at

    # =========================
    # 🔹 LAST RESULT
    # =========================
    last_result = records.filter(final_result__isnull=False).first()

    if last_result:
        last_result.result = last_result.final_result
        last_result.result_class = "negative" if last_result.final_result == "Negative" else "positive"
        last_result.date = last_result.created_at
        last_result.confidence = 90  # optional

    # =========================
    # 🔹 FINAL RESPONSE
    # =========================
    return render(request, "profile.html", {
        "patient": patient,
        "total_tests": total_tests,
        "completed_tests": completed_tests,
        "pending_tests": pending_tests,
        "recent_activities": recent_activities,
        "last_result": last_result,
        "show_header": False
    })

def patient_check(request):

    if not request.session.get('patient_id'):
        return redirect("login")

    result            = None
    redirect_url      = None
    confidence        = 0
    matched_symptoms  = []
    severity          = None
    recommended_test  = None
    description       = None
    gemini_response   = None
    selected_disease  = None

    json_path = os.path.join(settings.BASE_DIR, 'app', 'symptoms.json')

    with open(json_path, 'r') as file:
        data = json.load(file)

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "check":
            selected_symptoms = request.POST.getlist("symptoms")

            if not selected_symptoms:
                return render(request, "patient_check.html", {
                    "error": "Please select at least one symptom.",
                    "show_header": False
                })

            best_score = 0
            best_total_weight = 1

            for key, disease in data.items():
                disease_symptoms = disease["symptoms"]
                total_weight     = sum(s["weight"] for s in disease_symptoms)

                score        = 0
                temp_matched = []

                for s in disease_symptoms:
                    if s["name"] in selected_symptoms:
                        score += s["weight"]
                        temp_matched.append(s["name"])

                if score > best_score:
                    best_score         = score
                    best_total_weight  = total_weight
                    result             = disease["name"]
                    selected_disease   = disease
                    redirect_url       = disease["redirect_url"]

                    matched_symptoms = [
                        s.replace("_", " ").title()
                        for s in temp_matched
                    ]

                    severity         = disease.get("severity")
                    recommended_test = disease.get("recommended_test")
                    description      = disease.get("description")

            if best_total_weight > 0:
                confidence = round((best_score / best_total_weight) * 100, 2)

            # Store in session (only check, not save to DB)
            request.session['result']        = result
            request.session['redirect_url']  = redirect_url
            request.session['symptoms']      = selected_symptoms

            # 🔥 UPDATED GEMINI LOGIC (SHORT & HUMAN FRIENDLY)
            if selected_disease and confidence >= 40:
                base_prompt = selected_disease.get("gemini_prompt", "")

                readable_symptoms = ", ".join(
                    s.replace("_", " ").title() for s in selected_symptoms
                )

                final_prompt = (
                    f"A patient has these symptoms: {readable_symptoms}.\n"
                    f"The most likely condition is: {result}.\n\n"

                    f"Explain this in a very simple and friendly way like a doctor talking to a normal person.\n\n"

                    f"Rules:\n"
                    f"- Write ONLY 4 to 5 short lines\n"
                    f"- No headings\n"
                    f"- No bullet points\n"
                    f"- Use very simple English\n"
                    f"- Avoid medical jargon\n"
                    f"- Keep it clear and helpful\n\n"

                    f"Must include:\n"
                    f"- What the condition is\n"
                    f"- Why these symptoms happen\n"
                    f"- Basic care advice\n"
                    f"- When to see a doctor\n\n"

                    f"{base_prompt}"
                )

                gemini_response = get_gemini_response(final_prompt)

            elif confidence < 40:
                gemini_response = (
                    "⚠️ Symptoms are not clearly matching any condition. "
                    "Try selecting more accurate symptoms or consult a doctor."
                )

            return render(request, "patient_check.html", {
                "result"          : result,
                "redirect_url"    : redirect_url,
                "confidence"      : confidence,
                "matched_symptoms": matched_symptoms,
                "severity"        : severity,
                "recommended_test": recommended_test,
                "description"     : description,
                "gemini_response" : gemini_response,
                "show_header"     : False,
            })

    return render(request, "patient_check.html", {
        "show_header": False
    })

def save_report(request):
    if request.method == "POST":
        # 🔒 1. Session & Login Check
        if not request.session.get('patient_id'):
            return JsonResponse({"status": "error", "message": "Not logged in"})

        patient_id   = request.session.get('patient_id')
        result       = request.session.get('result')
        raw_url      = request.session.get('redirect_url')
        symptoms     = request.session.get('symptoms', [])

        if not result:
            return JsonResponse({"status": "error", "message": "No result found in session to save"})

        # ✅ 2. URL Mapping (Fixing inconsistent names before saving)
        url_map = {
            "malaria_predict": "malaria",
            "pneumonia_predict": "pneumonia"
        }
        redirect_url = url_map.get(raw_url, raw_url)

        # ✅ 3. Normalize symptoms
        symptoms_str = ", ".join(sorted(symptoms))

        # 🔍 4. SMART CHECK: 10-minute "De-duplication" window
        existing_report = PatientHistory.objects.filter(
            patient_id=patient_id,
            disease_name=result,
            symptoms=symptoms_str,
            created_at__gte=now() - timedelta(minutes=10) 
        ).first()

        if existing_report:
            # ✅ 5. UPDATE EXISTING: Refresh timestamp
            existing_report.created_at = now()
            existing_report.redirect_url = redirect_url # Ensure URL is also updated if changed
            existing_report.save()

            # 🔥 TRACKING: Link session to this ID
            request.session['current_report_id'] = existing_report.id

            return JsonResponse({
                "status": "success", 
                "message": "Recent report updated successfully"
            })
        
        else:
            # ✅ 6. CREATE NEW: Fresh entry for a new diagnostic session
            new_report = PatientHistory.objects.create(
                patient_id   = patient_id,
                disease_name = result,
                redirect_url = redirect_url,
                symptoms     = symptoms_str,
                final_result = None
            )

            # 🔥 TRACKING: Link session to the new ID
            request.session['current_report_id'] = new_report.id

            return JsonResponse({
                "status": "success", 
                "message": "New report saved successfully"
            })

    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)

def generate_ai_guidance(result, final_result):
    import re

    prompt = f"""
    A patient has been tested for {result}.
    Final result is: {final_result}.

    Give response STRICTLY in this format.

    Explanation:
    - point
    - point

    Diet:
    - Eat:
    - Avoid:

    Exercise:
    - point
    - point

    Precautions:
    - point
    - point

    Medicines:
    - point
    - point

    When to see a doctor:
    - point
    - point

    Rules:
    - ONLY bullet points
    - NO paragraphs
    - Minimum 2 points per section
    - Use simple English
    - Do not add extra text outside sections
    """

    try:
        response = get_gemini_response(prompt)

        print("AI RAW RESPONSE:", response)

        # 🚨 1. Handle empty / weak response
        if not response or len(response.strip().split("\n")) < 6:
            return get_fallback_response(result, final_result)

        # 🚨 2. CLEAN RESPONSE (VERY IMPORTANT)
        cleaned = response

        # remove markdown (*, **, ### etc.)
        cleaned = re.sub(r'\*{1,3}', '', cleaned)
        cleaned = re.sub(r'#{1,6}\s*', '', cleaned)

        # remove extra blank lines
        cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)

        # remove extra spaces
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)

        # trim
        cleaned = cleaned.strip()

        return cleaned

    except Exception as e:
        print("AI ERROR:", str(e))
        return get_fallback_response(result, final_result)


def get_fallback_response(result, final_result):
    return f"""Explanation:
- This result indicates {final_result.lower()} condition related to {result}
- It may require attention depending on symptoms

Diet:
- Eat: fresh fruits, vegetables, balanced diet
- Avoid: junk food, oily and sugary items

Exercise:
- Do light walking daily
- Try simple stretching or yoga

Precautions:
- Take proper rest
- Stay hydrated and avoid stress

Medicines:
- Paracetamol for mild symptoms (if needed)
- Always consult doctor before medication

When to see a doctor:
- If symptoms increase or do not improve
- If you feel pain, weakness, or discomfort
"""

def history(request):
    # 🔒 1. Check login
    patient_id = request.session.get('patient_id')
    if not patient_id:
        return redirect("login")

    # 🔍 2. Database-Level Filtering (Scalable & Professional)
    search_query  = request.GET.get('search', '').strip()
    result_filter = request.GET.get('result_filter', '').strip()

    # Start with all records for this patient
    query = Q(patient_id=patient_id)

    # Filter by Search (Disease Name or Symptoms) directly in DB
    if search_query:
        query &= (Q(disease_name__icontains=search_query) | Q(symptoms__icontains=search_query))

    # Filter by Result Status (Positive/Negative/Pending)
    if result_filter in ["Positive", "Negative"]:
        query &= Q(final_result=result_filter)
    elif result_filter == "Pending":
        query &= Q(final_result__isnull=True)

    # 🔥 3. Fetch Records (Latest First)
    # Since your Save Logic now updates existing records, duplicates are handled at the source.
    records_queryset = PatientHistory.objects.filter(query).order_by('-created_at')

    # 🎨 4. Enrich Data for UI
    for record in records_queryset:
        # Standardize Redirect URLs (Permanent fix logic)
        url_map = {
            "malaria_predict": "malaria",
            "pneumonia_predict": "pneumonia"
        }
        record.redirect_url = url_map.get(record.redirect_url, record.redirect_url or "dashboard")

        # Format Symptoms for Display
        if record.symptoms:
            record.symptoms_display = record.symptoms.replace("_", " ").title()

        # Formatting Dates
        record.formatted_date = record.created_at.strftime("%d %b %Y")

        # Result & Status Classes (Kept as per your request)
        if record.final_result == "Positive":
            record.result_label, record.result_class = "Positive", "badge-positive"
            record.status, record.status_class = "Completed", "badge-completed"
        elif record.final_result == "Negative":
            record.result_label, record.result_class = "Negative", "badge-negative"
            record.status, record.status_class = "Completed", "badge-completed"
        else:
            record.result_label, record.result_class = "Pending", "badge-pending"
            record.status, record.status_class = "Pending", "badge-pending"

    # 📄 5. Pagination (10 per page)
    paginator = Paginator(records_queryset, 10)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    return render(request, "history.html", {
        "page_obj": page_obj,
        "records": page_obj, 
        "search_query": search_query,
        "result_filter": result_filter,
        "show_header": False,
    })

def check_test(request):
    tests = [
        {"name": "Diabetes", "url": "diabetes_predict"},
        {"name": "Heart Disease", "url": "heart_predict"},
        {"name": "Kidney Disease", "url": "kidney_predict"},
        {"name": "Liver Disease", "url": "liver_predict"},
        {"name": "Malaria", "url": "malaria"},
        {"name": "Pneumonia", "url": "pneumonia"},
    ]

    return render(request, "check_test.html", {"tests": tests})

def redirect_test(request, test_name):
    if test_name == "diabetes":
        return redirect('diabetes_predict')

    elif test_name == "heart":
        return redirect('heart_predict')

    elif test_name == "kidney":
        return redirect('kidney_predict')

    elif test_name == "liver":
        return redirect('liver_predict')

    elif test_name == "malaria":
        return redirect('malaria')

    elif test_name == "pneumonia":
        return redirect('pneumonia')

    else:
        return redirect('check_test')
    
def analytics(request):

    # 🔒 Login check
    if not request.session.get('patient_id'):
        return redirect("login")

    patient_id = request.session.get('patient_id')

    # 🔹 Get all records
    records = PatientHistory.objects.filter(
        patient_id=patient_id
    ).order_by('-created_at')

    # =========================
    # 🔹 OVERVIEW DATA
    # =========================
    total_tests = records.count()
    positive_count = records.filter(final_result="Positive").count()
    negative_count = records.filter(final_result="Negative").count()

    last_result = records.filter(final_result__isnull=False).first()

    # =========================
    # 🔹 HEALTH SCORE
    # =========================
    health_score = 100 - (positive_count * 10)
    health_score = max(0, health_score)

    # =========================
    # 🔹 DISEASE DISTRIBUTION
    # =========================
    disease_counts = {}

    for r in records:
        name = r.disease_name or "Unknown"
        disease_counts[name] = disease_counts.get(name, 0) + 1

    chart_labels = list(disease_counts.keys())
    chart_values = list(disease_counts.values())

    # =========================
    # 🔹 RESULT ANALYSIS
    # =========================
    result_labels = ["Positive", "Negative"]
    result_values = [positive_count, negative_count]

    # =========================
    # 🔹 TIME TREND
    # =========================
    trend_labels = []
    trend_values = []

    for r in records[:10][::-1]:
        trend_labels.append(r.created_at.strftime("%d %b"))
        trend_values.append(1 if r.final_result == "Positive" else 0)

    # =========================
    # 🔹 MODEL ACCURACY (STATIC)
    # =========================
    model_labels = ["Diabetes", "Heart", "Kidney", "Liver", "Malaria", "Pneumonia"]
    model_values = [85, 82, 90, 88, 95, 93]  # 👉 Replace with your real values

    # =========================
    # 🔹 RECENT ACTIVITY
    # =========================
    recent = records[:5]

    return render(request, "analytics.html", {
        "total_tests": total_tests,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "last_result": last_result,
        "health_score": health_score,

        "chart_labels": chart_labels,
        "chart_values": chart_values,

        "result_labels": result_labels,
        "result_values": result_values,

        "trend_labels": trend_labels,
        "trend_values": trend_values,

        "model_labels": model_labels,
        "model_values": model_values,

        "recent": recent,
        "show_header": False
    })


# 🔹 LOGOUT VIEW
def logout_view(request):
    request.session.flush()
    messages.success(request, "Logged out successfully!")
    return redirect("/")


def index(request):
    return render(request, 'index.html', {'show_header': True})

def diabetes_predict(request):
    # 🔒 1. Check login
    if not request.session.get('patient_id'):
        return redirect("login")

    if request.method == 'POST':
        try:
            # 🧪 2. Collect Data from Form (SAFE parsing)
            data = [
                int(request.POST.get('preg', 0)),
                float(request.POST.get('glucose', 0)),
                float(request.POST.get('bp', 0)),
                float(request.POST.get('skin', 0)),
                float(request.POST.get('insulin', 0)),
                float(request.POST.get('bmi', 0)),
                float(request.POST.get('dpf', 0)),
                int(request.POST.get('age', 0))
            ]

            print("INPUT DATA:", data)  # 🔍 Debug

            # 🧠 3. Run ML Prediction
            prediction = diabetes_model.predict([data])[0]
            probability = diabetes_model.predict_proba([data])[0][1]

            # 🎯 4. Result Logic
            if prediction == 1:
                result = "Diabetes Detected ❌"
                final_result = "Positive"
            else:
                result = "No Diabetes ✅"
                final_result = "Negative"

            print("ML RESULT:", result, final_result, probability)

            # 🧠 5. Generate AI Guidance (SAFE)
            ai_advice = generate_ai_guidance(result, final_result)

            # 🔍 Debug AI
            print("AI ADVICE:", ai_advice)

            # 🔥 6. Update Existing Report (if exists)
            report_id = request.session.get('current_report_id')

            if report_id:
                record = PatientHistory.objects.filter(id=report_id).first()

                if record:
                    record.final_result = final_result
                    record.probability = round(probability * 100, 2)
                    record.ai_advice = ai_advice  # ✅ SAVE AI RESPONSE
                    record.save()

                # 🧹 CLEAR SESSION (VERY IMPORTANT)
                request.session.pop('current_report_id', None)

            # 📊 7. Render Result Page
            return render(request, 'result.html', {
                'result': result,
                'final_result': final_result,
                'probability': round(probability * 100, 2),
                'ai_advice': ai_advice
            })

        except Exception as e:
            print("ERROR IN DIABETES PREDICT:", str(e))

            # 🚨 Fail-safe UI (never break page)
            return render(request, 'result.html', {
                'result': "Error occurred",
                'final_result': "Unknown",
                'probability': 0,
                'ai_advice': get_fallback_response("Diabetes", "Unknown")
            })

    # 📄 GET request
    return render(request, 'diabetes.html')


def heart_predict(request):
    # 🔒 1. Check login
    if not request.session.get('patient_id'):
        return redirect("login")

    if request.method == "POST":
        try:
            # 🧪 2. Collect Data (SAFE)
            data = {
                'age': int(request.POST.get('age', 0)),
                'sex': int(request.POST.get('sex', 0)),
                'cp': int(request.POST.get('cp', 0)),
                'trestbps': int(request.POST.get('trestbps', 0)),
                'chol': int(request.POST.get('chol', 0)),
                'fbs': int(request.POST.get('fbs', 0)),
                'restecg': int(request.POST.get('restecg', 0)),
                'thalach': int(request.POST.get('thalach', 0)),
                'exang': int(request.POST.get('exang', 0)),
                'oldpeak': float(request.POST.get('oldpeak', 0)),
                'slope': int(request.POST.get('slope', 0)),
                'ca': int(request.POST.get('ca', 0)),
                'thal': int(request.POST.get('thal', 0)),
            }

            print("HEART INPUT:", data)  # 🔍 Debug

            input_df = pd.DataFrame([data])

            # 🧠 3. Prediction
            prediction = heart_model.predict(input_df)[0]
            probability = heart_model.predict_proba(input_df)[0][1]

            # 🎯 4. Result Logic
            if prediction == 1:
                result = "Heart Disease Detected ❌"
                final_result = "Positive"
            else:
                result = "No Heart Disease ✅"
                final_result = "Negative"

            # 📊 5. Confidence Logic (🔥 IMPROVED)
            confidence = round(probability * 100, 2)

            if confidence >= 80:
                confidence_level = "High"
            elif confidence >= 60:
                confidence_level = "Moderate"
            else:
                confidence_level = "Low"

            print("HEART RESULT:", result, final_result, confidence)

            # 🧠 6. AI Guidance
            ai_advice = generate_ai_guidance(result, final_result)

            print("AI ADVICE:", ai_advice)

            # 🔥 7. Update DB
            report_id = request.session.get('current_report_id')

            if report_id:
                record = PatientHistory.objects.filter(id=report_id).first()

                if record:
                    record.final_result = final_result
                    record.probability = confidence
                    record.ai_advice = ai_advice
                    record.save()

                # 🧹 Clear session safely
                request.session.pop('current_report_id', None)

            # 📊 8. Render Result
            context = {
                "model_name": "Heart Disease Prediction",
                "result": result,
                "probability": confidence,
                "confidence_level": confidence_level,  # 🔥 NEW
                "final_result": final_result,
                "ai_advice": ai_advice
            }

            return render(request, "result.html", context)

        except Exception as e:
            print("ERROR IN HEART PREDICT:", str(e))

            # 🚨 Fail-safe UI
            return render(request, "result.html", {
                "model_name": "Heart Disease Prediction",
                "result": "Error occurred",
                "final_result": "Unknown",
                "probability": 0,
                "confidence_level": "Low",
                "ai_advice": get_fallback_response("Heart Disease", "Unknown")
            })

    return render(request, "heart.html")

def kidney_predict(request):
    """
    Kidney Disease Prediction View (PRO VERSION)
    """

    # 🔒 1. Check login
    if not request.session.get('patient_id'):
        return redirect("login")

    if request.method == "POST":
        try:
            # 🧪 2. Collect Data (SAFE)
            sample = {
                'id': int(request.POST.get('id', 0)),
                'age': int(request.POST.get('age', 0)),
                'bp': int(request.POST.get('bp', 0)),
                'sg': float(request.POST.get('sg', 1.020)),
                'al': int(request.POST.get('al', 0)),
                'su': int(request.POST.get('su', 0)),
                'rbc': request.POST.get('rbc', 'normal'),
                'pc': request.POST.get('pc', 'normal'),
                'pcc': request.POST.get('pcc', 'notpresent'),
                'ba': request.POST.get('ba', 'notpresent'),
                'bgr': float(request.POST.get('bgr', 100)),
                'bu': float(request.POST.get('bu', 20)),
                'sc': float(request.POST.get('sc', 1.0)),
                'sod': float(request.POST.get('sod', 140)),
                'pot': float(request.POST.get('pot', 4.0)),
                'hemo': float(request.POST.get('hemo', 14)),
                'pcv': float(request.POST.get('pcv', 42)),
                'wc': float(request.POST.get('wc', 8000)),
                'rc': float(request.POST.get('rc', 5.0)),
                'htn': request.POST.get('htn', 'no'),
                'dm': request.POST.get('dm', 'no'),
                'cad': request.POST.get('cad', 'no'),
                'appet': request.POST.get('appet', 'good'),
                'pe': request.POST.get('pe', 'no'),
                'ane': request.POST.get('ane', 'no'),
            }

            print("KIDNEY INPUT:", sample)  # 🔍 Debug

            df = pd.DataFrame([sample])

            # 🧠 3. Prediction
            pred = kidney_model.predict(df)[0]
            proba = kidney_model.predict_proba(df)[0]

            # 🎯 4. Result Logic
            if pred == 1:
                result = "Kidney Disease Detected ❌"
                final_result = "Positive"
            else:
                result = "No Kidney Disease ✅"
                final_result = "Negative"

            # 📊 5. Confidence System (🔥 NEW)
            confidence = round(proba[1] * 100, 2)

            if confidence >= 80:
                confidence_level = "High"
            elif confidence >= 60:
                confidence_level = "Moderate"
            else:
                confidence_level = "Low"

            print("KIDNEY RESULT:", result, final_result, confidence)

            # 🧠 6. AI Guidance
            ai_advice = generate_ai_guidance(result, final_result)

            print("AI ADVICE:", ai_advice)

            # 🔥 7. Update DB
            report_id = request.session.get('current_report_id')

            if report_id:
                record = PatientHistory.objects.filter(id=report_id).first()

                if record:
                    record.final_result = final_result
                    record.probability = confidence
                    record.ai_advice = ai_advice
                    record.save()

                # 🧹 Safe session clear
                request.session.pop('current_report_id', None)

            # 📊 8. Render Result
            return render(
                request,
                "result.html",
                {
                    "model_name": "Kidney Disease Prediction",
                    "result": result,
                    "probability": confidence,
                    "confidence_level": confidence_level,
                    "final_result": final_result,
                    "ai_advice": ai_advice
                }
            )

        except Exception as e:
            print("ERROR IN KIDNEY PREDICT:", str(e))

            # 🚨 Fail-safe UI
            return render(
                request,
                "result.html",
                {
                    "model_name": "Kidney Disease Prediction",
                    "result": "Error occurred during prediction",
                    "final_result": "Unknown",
                    "probability": 0,
                    "confidence_level": "Low",
                    "ai_advice": get_fallback_response("Kidney Disease", "Unknown")
                }
            )

    return render(request, "predict.html")


def liver_predict(request):
    """
    Liver Disease Prediction View (PRO VERSION)
    """

    # 🔒 1. Check login
    if not request.session.get('patient_id'):
        return redirect("login")

    if request.method == "POST":
        try:
            # 🧪 2. Collect Data (SAFE)
            data = {
                'age': float(request.POST.get('age', 0)),
                'gender': request.POST.get('gender', 'Male'),
                'total_bilirubin': float(request.POST.get('total_bilirubin', 0)),
                'direct_bilirubin': float(request.POST.get('direct_bilirubin', 0)),
                'alkaline_phosphotase': float(request.POST.get('alkaline_phosphotase', 0)),
                'alamine_aminotransferase': float(request.POST.get('alamine_aminotransferase', 0)),
                'aspartate_aminotransferase': float(request.POST.get('aspartate_aminotransferase', 0)),
                'total_protiens': float(request.POST.get('total_protiens', 0)),
                'albumin': float(request.POST.get('albumin', 0)),
                'albumin_and_globulin_ratio': float(request.POST.get('albumin_and_globulin_ratio', 0)),
            }

            print("LIVER INPUT:", data)  # 🔍 Debug

            df = pd.DataFrame([data])

            # 🧠 3. Prediction
            prediction = liver_model.predict(df)[0]
            probability = liver_model.predict_proba(df)[0][1]

            # 🎯 4. Result Logic
            if prediction == 1:
                result = "Liver Disease Detected ❌"
                final_result = "Positive"
            else:
                result = "No Liver Disease ✅"
                final_result = "Negative"

            # 📊 5. Confidence System (🔥 NEW)
            confidence = round(probability * 100, 2)

            if confidence >= 80:
                confidence_level = "High"
            elif confidence >= 60:
                confidence_level = "Moderate"
            else:
                confidence_level = "Low"

            print("LIVER RESULT:", result, final_result, confidence)

            # 🧠 6. AI Guidance
            ai_advice = generate_ai_guidance(result, final_result)

            print("AI ADVICE:", ai_advice)

            # 🔥 7. Update DB
            report_id = request.session.get('current_report_id')

            if report_id:
                record = PatientHistory.objects.filter(id=report_id).first()

                if record:
                    record.final_result = final_result
                    record.probability = confidence
                    record.ai_advice = ai_advice
                    record.save()

                # 🧹 Safe session clear
                request.session.pop('current_report_id', None)

            # 📊 8. Render Result
            return render(request, "result.html", {
                "model_name": "Liver Disease Prediction",
                "result": result,
                "probability": confidence,
                "confidence_level": confidence_level,
                "final_result": final_result,
                "ai_advice": ai_advice
            })

        except Exception as e:
            print("ERROR IN LIVER PREDICT:", str(e))

            # 🚨 Fail-safe UI
            return render(request, "result.html", {
                "model_name": "Liver Disease Prediction",
                "result": "Error occurred during prediction",
                "final_result": "Unknown",
                "probability": 0,
                "confidence_level": "Low",
                "ai_advice": get_fallback_response("Liver Disease", "Unknown")
            })

    return render(request, "liver.html")


from django.core.files.storage import FileSystemStorage
def malaria_predict(request):
    """
    Malaria Detection View (PRO VERSION)
    """

    # 🔒 1. Check login
    if not request.session.get('patient_id'):
        return redirect("login")

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # 🔹 2. SAVE IMAGE
            user_img = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(user_img.name, user_img)
            file_url = fs.url(filename)
            full_path = fs.path(filename)

            print("IMAGE SAVED:", full_path)

            # 🔹 3. PREPROCESS IMAGE
            test_img = image.load_img(full_path, target_size=(224, 224))
            img_array = image.img_to_array(test_img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # 🔹 4. MODEL PREDICTION
            prediction = malaria_model.predict(img_array)

            raw_score = prediction[0][0]

            if raw_score > 0.5:
                final_result = "Negative"
                display_result = "Uninfected"
                confidence = round(raw_score * 100, 2)
            else:
                final_result = "Positive"
                display_result = "Parasitized (Malaria Detected)"
                confidence = round((1 - raw_score) * 100, 2)

            # 📊 5. Confidence Level (🔥 NEW)
            if confidence >= 80:
                confidence_level = "High"
            elif confidence >= 60:
                confidence_level = "Moderate"
            else:
                confidence_level = "Low"

            print("MALARIA RESULT:", display_result, final_result, confidence)

            # 🧠 6. AI Guidance
            ai_advice = generate_ai_guidance(display_result, final_result)

            print("AI ADVICE:", ai_advice)

            # 🔥 7. Update DB
            report_id = request.session.get('current_report_id')

            if report_id:
                record = PatientHistory.objects.filter(id=report_id).first()

                if record:
                    record.final_result = final_result
                    record.probability = confidence
                    record.ai_advice = ai_advice
                    record.save()

                # 🧹 Safe clear
                request.session.pop('current_report_id', None)

            # 🔹 8. RETURN RESULT PAGE (🔥 IMPORTANT CHANGE)
            return render(request, 'result.html', {
                'model_name': "Malaria Detection",
                'result': display_result,
                'probability': confidence,
                'confidence_level': confidence_level,
                'image_url': file_url,
                'final_result': final_result,
                'ai_advice': ai_advice
            })

        except Exception as e:
            print("ERROR IN MALARIA:", str(e))

            # 🚨 Fail-safe UI
            return render(request, 'result.html', {
                'model_name': "Malaria Detection",
                'result': "Error processing image",
                'final_result': "Unknown",
                'probability': 0,
                'confidence_level': "Low",
                'ai_advice': get_fallback_response("Malaria", "Unknown")
            })

    return render(request, 'malaria.html')
def pneumonia_predict(request):
    """
    Pneumonia Detection View (PRO VERSION)
    """

    # 🔒 1. Check login
    if not request.session.get('patient_id'):
        return redirect("login")

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # 🔹 2. SAVE IMAGE
            user_img = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(user_img.name, user_img)
            file_url = fs.url(filename)
            full_path = fs.path(filename)

            print("PNEUMONIA IMAGE:", full_path)

            # 🔹 3. LOAD & PREPROCESS IMAGE
            try:
                img = Image.open(full_path).convert("RGB")
            except Exception as e:
                print("IMAGE ERROR:", str(e))
                return render(request, 'result.html', {
                    'model_name': "Pneumonia Detection",
                    'result': "Invalid Image File",
                    'final_result': "Unknown",
                    'probability': 0,
                    'confidence_level': "Low",
                    'ai_advice': get_fallback_response("Pneumonia", "Unknown")
                })

            img = pneumonia_transform(img)
            img = img.unsqueeze(0).to(device)

            # 🔹 4. MODEL PREDICTION (PyTorch)
            with torch.no_grad():
                outputs = pneumonia_model(img)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            classes = ['NORMAL', 'PNEUMONIA']
            display_result = classes[pred]
            confidence = round(probs[0][pred].item() * 100, 2)

            # 🎯 5. Result Mapping
            if display_result == "PNEUMONIA":
                final_result = "Positive"
            else:
                final_result = "Negative"

            # 📊 6. Confidence Level (🔥 NEW)
            if confidence >= 80:
                confidence_level = "High"
            elif confidence >= 60:
                confidence_level = "Moderate"
            else:
                confidence_level = "Low"

            print("PNEUMONIA RESULT:", display_result, final_result, confidence)

            # 🧠 7. AI Guidance
            ai_advice = generate_ai_guidance(display_result, final_result)

            print("AI ADVICE:", ai_advice)

            # 🔥 8. Update DB
            report_id = request.session.get('current_report_id')

            if report_id:
                record = PatientHistory.objects.filter(id=report_id).first()

                if record:
                    record.final_result = final_result
                    record.probability = confidence
                    record.ai_advice = ai_advice
                    record.save()

                # 🧹 Safe session clear
                request.session.pop('current_report_id', None)

            # 🔹 9. RETURN RESULT PAGE (🔥 FIXED)
            return render(request, 'result.html', {
                'model_name': "Pneumonia Detection",
                'result': display_result,
                'probability': confidence,
                'confidence_level': confidence_level,
                'image_url': file_url,
                'final_result': final_result,
                'ai_advice': ai_advice
            })

        except Exception as e:
            print("ERROR IN PNEUMONIA:", str(e))

            # 🚨 Fail-safe UI
            return render(request, 'result.html', {
                'model_name': "Pneumonia Detection",
                'result': "Error processing image",
                'final_result': "Unknown",
                'probability': 0,
                'confidence_level': "Low",
                'ai_advice': get_fallback_response("Pneumonia", "Unknown")
            })

    return render(request, 'pneumonia.html')