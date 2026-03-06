FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends git nodejs && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory /workspace

WORKDIR /workspace

# Install the agent package and all its dependencies
COPY . .
RUN pip install --no-cache-dir -e .

CMD ["bash"]
