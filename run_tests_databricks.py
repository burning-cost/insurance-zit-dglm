"""
Script to run ZIT-DGLM tests on Databricks via Jobs API (serverless compute).
"""
import os
import sys
import time
import base64
import subprocess

# Load credentials
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

workspace_path = "/Workspace/insurance-zit-dglm"

# Upload the project files to workspace
print("Uploading project to Databricks workspace...")
result = subprocess.run(
    [
        "databricks", "workspace", "import-dir",
        "/home/ralph/repos/insurance-zit-dglm",
        workspace_path,
        "--overwrite",
    ],
    capture_output=True,
    text=True,
    env=os.environ,
)
last_lines = result.stdout.strip().split("\n")[-5:]
print("\n".join(last_lines))
if result.returncode != 0:
    print("Upload STDERR:", result.stderr[:300])
print("Upload complete.")

# Write test runner notebook using SDK
notebook_content = r"""# Databricks notebook source
# MAGIC %pip install catboost polars numpy scipy matplotlib pytest --quiet

# COMMAND ----------

import subprocess, sys

# Install the package in editable mode
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-zit-dglm", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-1000:] if result.stdout else "")
print(result.stderr[-500:] if result.stderr else "")

# COMMAND ----------

# Run pytest
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-zit-dglm/tests/",
        "-v",
        "--tb=short",
        "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-zit-dglm",
)

print(result.stdout[-8000:])
print(result.stderr[-2000:])

if result.returncode != 0:
    raise Exception(f"Tests FAILED (exit code {result.returncode})")
else:
    print("\n=== ALL TESTS PASSED ===")
"""

nb_workspace_path = f"{workspace_path}/run_tests_zit"
encoded_content = base64.b64encode(notebook_content.encode("utf-8")).decode("utf-8")

w.workspace.import_(
    path=nb_workspace_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=encoded_content,
    overwrite=True,
)
print(f"Test notebook written to {nb_workspace_path}")

# Submit as a one-time job run using serverless compute
print("Submitting test job (serverless)...")
run = w.jobs.submit(
    run_name="insurance-zit-dglm-tests",
    tasks=[
        jobs.SubmitTask(
            task_key="run_tests",
            notebook_task=jobs.NotebookTask(
                notebook_path=nb_workspace_path,
            ),
            # Serverless: no cluster spec needed
            environment_key=None,
        )
    ],
)
run_id = run.run_id
print(f"Run submitted: run_id={run_id}")
print(f"Tracking URL: {os.environ['DATABRICKS_HOST']}#job/run/{run_id}")

# Poll for completion
print("Waiting for completion...")
while True:
    run_state = w.jobs.get_run(run_id=run_id)
    state = run_state.state
    lc = state.life_cycle_state
    print(f"  [{lc}] {state.state_message or ''}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(20)

# Get result
result_state = run_state.state.result_state
print(f"\nFinal state: {result_state}")

# Get notebook output
tasks = run_state.tasks
if tasks:
    task = tasks[0]
    output = w.jobs.get_run_output(run_id=task.run_id)
    if output.notebook_output:
        print("\n=== NOTEBOOK OUTPUT ===")
        print(output.notebook_output.result)

if str(result_state) == "ResultState.SUCCESS":
    print("\nTests PASSED on Databricks.")
    sys.exit(0)
else:
    print("\nTests FAILED on Databricks.")
    sys.exit(1)
