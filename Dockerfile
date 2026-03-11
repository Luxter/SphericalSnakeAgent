FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git nodejs npm \
    libcairo2-dev libpango1.0-dev libjpeg62-turbo-dev libgif-dev libpixman-1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory /workspace

WORKDIR /workspace

# Install the agent package and all its dependencies
COPY . .
RUN pip install --no-cache-dir -e .

# Install Node.js tools dependencies (canvas + gifencoder for render_video.js)
RUN npm install --prefix /workspace/tools --omit=dev

CMD ["bash"]
