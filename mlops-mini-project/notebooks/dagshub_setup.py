# -*- coding: utf-8 -*-
"""Initialise MLflow tracking with DagsHub for the Tweet Emotion project.

Run this helper script once to verify the connection, or import the setup
block into any experiment notebook before calling ``mlflow.start_run()``.

Requirements:
    pip install mlflow dagshub
"""

import mlflow
import dagshub

# Configure the remote MLflow tracking server hosted on DagsHub
mlflow.set_tracking_uri("https://dagshub.com/DeepuML/Mlops-Mini-Project.mlflow")

# Initialise DagsHub integration (also enables MLflow autologging hooks)
dagshub.init(repo_owner="DeepuML", repo_name="Mlops-Mini-Project", mlflow=True)

# Smoke-test: log a dummy run to confirm the connection is working
mlflow.set_experiment("DagsHub Connection Test")
with mlflow.start_run(run_name="connection-check"):
    mlflow.log_param("status", "connected")
    mlflow.log_metric("test_metric", 1.0)
    print("DagsHub MLflow tracking configured successfully.")
    print("Tracking URI:", mlflow.get_tracking_uri())
