#!/bin/bash

source .venv/bin/activate

celery -A lab_tools worker -P solo -l info