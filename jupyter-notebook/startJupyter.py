#!/usr/bin/env python3
import os, subprocess, sys

workload_port = os.environ.get("workload_port", "7008")
workload_id = os.environ.get("workload_id", "wl_unknown")
# When running behind an nginx subpath proxy, JUPYTER_BASE_URL is set by the
# aistudio-server worker (e.g. /jupyter/T4/jup-20260729-xxx/).
# Without this, JupyterLab serves all assets from / and breaks behind a subpath.
base_url = os.environ.get("JUPYTER_BASE_URL", "/")

# Who may open the notebook: Jupyter reads the token from JUPYTER_TOKEN in the
# environment (AI Studio sets it, one per notebook, and hands it only to the
# notebook's owner). With no JUPYTER_TOKEN, Jupyter makes up a random one and
# prints it in its log, so the server never runs without a password. The token
# is not passed on the command line, where any process on the node could read it.
jupyter_cmd = [
    "jupyter", "notebook",
    "--no-browser",
    "--ip=0.0.0.0",
    f"--port={workload_port}",
    "--allow-root",
    f"--ServerApp.base_url={base_url}",
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
