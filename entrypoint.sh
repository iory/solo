#!/bin/bash

uvicorn --app-dir=. main:app --port 8080 --host 0.0.0.0 --workers 1
