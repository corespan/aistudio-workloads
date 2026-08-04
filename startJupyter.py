#!/usr/bin/env python3
import os, subprocess, sys

try:
    from aiAgent.messageQueue import push_to_mq, create_msg
    _AIAGENT = True
except ImportError:
    _AIAGENT = False

workload_port = os.environ.get("workload_port", "7008")
workload_id = os.environ.get("workload_id", "wl_unknown")
machine_id = os.environ.get("machine_id", "machine_unknown")
# When running behind an nginx subpath proxy, JUPYTER_BASE_URL is set by the
# aistudio-server worker (e.g. /jupyter/T4/jup-20260729-xxx/).
# Without this, JupyterLab serves all assets from / and breaks behind a subpath.
base_url = os.environ.get("JUPYTER_BASE_URL", "/")

jupyter_cmd = [
    "jupyter", "notebook",
    "--no-browser",
    "--ip=0.0.0.0",
    f"--port={workload_port}",
    "--NotebookApp.token=",
    "--NotebookApp.password=''",
    "--allow-root",
    f"--ServerApp.base_url={base_url}",
    "--ServerApp.allow_origin=*",       # required when served via HTTPS reverse proxy
    "--ServerApp.disable_check_xsrf=True",  # XSRF cookie check breaks behind HTTPS proxy
]

def push_error(reason):
    if _AIAGENT:
        msg = create_msg(workload_id, machine_id, "status", "Internal Workload Error", reason)
        push_to_mq(msg)
    else:
        print(f"ERROR: {reason}", flush=True)


if __name__ == "__main__":
    try:
        work_dir = f"/data/{workload_id}"
        os.makedirs(work_dir, exist_ok=True)
        
        proc = subprocess.Popen(jupyter_cmd, stdout=sys.stdout, stderr=sys.stderr,cwd=work_dir)
        rc = proc.wait()
        if rc != 0:
            push_error(f"Jupyter exited with code {rc}")
        sys.exit(rc)
    except Exception as e:
        push_error(f"Exception while starting Jupyter: {e}")
        raise
