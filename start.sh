#!/bin/bash
web: gunicorn -w 1 -t 300 app:app

