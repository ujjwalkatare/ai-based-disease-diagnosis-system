#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

import os
import sys
import warnings
import logging


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

warnings.filterwarnings('ignore')            
logging.getLogger('tensorflow').setLevel(logging.ERROR)  


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_based_disease_diagnosis.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError("Couldn't import Django.") from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()