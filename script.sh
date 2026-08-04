#!/bin/bash
# Install aiAgent only when AIAGENT_VERSION is provided (aiWorkbench flow).
# Skipped when launched from aistudio-server which has no Nexus credentials.
PACKAGE_REGISTRY="${PACKAGE_REGISTRY:-nexus}"

if [ -n "${AIAGENT_VERSION}" ]; then
    if [ "$PACKAGE_REGISTRY" = "nexus" ]; then
        echo "Installing aiAgent from Nexus..."
        HOST_IP="${NEXUSREPO_REGISTRY_URL%%:*}"
        pip install https://${NEXUSREPO_ROUSER}:${NEXUSREPO_ROPASSWORD}@${HOST_IP}/repository/workbench-utils/packages/aiagent/${AIAGENT_VERSION}/aiAgent-${AIAGENT_VERSION}-py3-none-any.whl --trusted-host ${HOST_IP}
    else
        echo "Installing aiAgent from GCP Artifact Registry..."
        pip install keyrings.google-artifactregistry-auth
        pip install aiAgent==${AIAGENT_VERSION} --extra-index-url https://${GCP_PYTHON_REGISTRY_URL:-us-python.pkg.dev}/${GCP_PROJECT_ID:-aimlworkbench}/workbench-utils/simple/
    fi
else
    echo "AIAGENT_VERSION not set — skipping aiAgent install"
fi

workload_port=${workload_port:-7008}

mkdir -p /data/${workload_id}
cp ./notebooks/*.ipynb /data/${workload_id}/

CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
python3 ./startJupyter.py