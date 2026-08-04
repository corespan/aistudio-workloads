#!/usr/bin/env python3
import os, subprocess, sys

workload_port = os.environ.get("workload_port", "7008")
workload_id = os.environ.get("workload_id", "wl_unknown")
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


if __name__ == "__main__":
    try:
        work_dir = f"/data/{workload_id}"
        os.makedirs(work_dir, exist_ok=True)

        proc = subprocess.Popen(jupyter_cmd, stdout=sys.stdout, stderr=sys.stderr, cwd=work_dir)
        rc = proc.wait()
        if rc != 0:
            print(f"ERROR: Jupyter exited with code {rc}", flush=True)
        sys.exit(rc)
    except Exception as e:
        print(f"ERROR: Exception while starting Jupyter: {e}", flush=True)
        raise
