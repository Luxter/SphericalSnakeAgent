FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory /workspace

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source so the image can be used standalone (volumes override at runtime)
COPY . .

CMD ["bash"]
