# SoftGNN agent API for InteractiveAI.
#
#   docker compose up --build                  (uses .env, see .env.example)
# or
#   docker build -t softgnn-agent-api .
#   docker run -p 8000:8000 -e API_TOKEN=<token> softgnn-agent-api

# lightsim2grid publishes no Linux arm64 wheels, so the image is always built
# for amd64. Apple Silicon machines run it through emulation.
FROM --platform=linux/amd64 python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# The Grid2Op environment InteractiveAI runs, from a tested commit of
# AI4REALNET/grid2op-scenario. git is only needed for this step.
ARG GRID2OP_SCENARIO_REF=8b6167f9411b592f0eabd286de991ec63986b6d9
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && git init -q /tmp/grid2op-scenario \
 && git -C /tmp/grid2op-scenario fetch -q --depth 1 \
        https://github.com/AI4REALNET/grid2op-scenario.git "${GRID2OP_SCENARIO_REF}" \
 && git -C /tmp/grid2op-scenario checkout -q FETCH_HEAD \
 && mkdir -p /root/data_grid2op \
 && cp -r /tmp/grid2op-scenario/ai4realnet_small /root/data_grid2op/ai4realnet_small \
 && rm -rf /tmp/grid2op-scenario \
 && apt-get purge -y --auto-remove git \
 && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch goes in first: it satisfies the torch pin in
# requirements_docker.txt, so pip never pulls the much larger CUDA build.
COPY requirements_docker.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 \
 && pip install -r requirements_docker.txt

# Agent code and released artifacts
COPY GNNAgent.py ./
COPY gnn/ ./gnn/
COPY evaluation/utilities.py ./evaluation/
COPY app/ ./app/
COPY data/best_model/ ./data/best_model/
COPY data/actions/soft_actions.npy ./data/actions/
COPY data/scaler_all.pkl ./data/

ENV GRID2OP_ENV=/root/data_grid2op/ai4realnet_small

EXPOSE 8000

# Pass API_TOKEN at runtime: docker run -e API_TOKEN=<your_token> ...
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
