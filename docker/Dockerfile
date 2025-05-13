FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


# Install python
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get -y install --no-install-recommends python3.10-dev python3-pip libgl-dev libx11-dev libglib2.0-0

# Set default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /workspace

# Install uv
RUN pip install uv

# Install main package
COPY pyproject.toml .
COPY uv.lock .

# Copy scripts
COPY src/ src/
COPY scripts/ scripts/
COPY checkpoints/ checkpoints/

RUN uv sync --frozen

ENTRYPOINT ["/bin/bash", "-c"]