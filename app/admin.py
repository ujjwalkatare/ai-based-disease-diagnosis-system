from django.contrib import admin
from .models import Patient


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):

    # 🔹 Columns shown in admin list page
    list_display = (
        'id',
        'name',
        'email',
        'mobile',
        'aadhaar',
        'age',
        'gender',
        'created_at'
    )

    # 🔹 Search functionality
    search_fields = (
        'name',
        'email',
        'mobile',
        'aadhaar'
    )

    # 🔹 Filters (right side)
    list_filter = (
        'gender',
        'created_at'
    )

    # 🔹 Sorting (latest first)
    ordering = ('-created_at',)

    # 🔹 Read-only fields (cannot edit)
    readonly_fields = ('created_at',)

    # 🔹 Better form layout in admin
    fieldsets = (
        ("Basic Information", {
            'fields': ('name', 'email', 'mobile', 'aadhaar')
        }),

        ("Authentication", {
            'fields': ('password',)
        }),

        ("Personal Info", {
            'fields': ('age', 'gender')
        }),

        ("System Info", {
            'fields': ('created_at',)
        }),
    )