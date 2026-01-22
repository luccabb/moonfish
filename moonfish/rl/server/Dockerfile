FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install moonfish from PyPI (includes chess dependency)
RUN pip install --no-cache-dir moonfish

# Install additional dependencies for the RL server
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    "uvicorn[standard]>=0.23.0" \
    httpx>=0.24.0 \
    pydantic>=2.0.0

# Copy the RL module into the installed moonfish package
# First, find where moonfish is installed
RUN MOONFISH_PATH=$(python -c "import moonfish; import os; print(os.path.dirname(moonfish.__file__))") && \
    mkdir -p ${MOONFISH_PATH}/rl/server

# Copy RL module files - we need to do this in a separate step
COPY __init__.py /tmp/rl/__init__.py
COPY models.py /tmp/rl/models.py
COPY client.py /tmp/rl/client.py
COPY server/__init__.py /tmp/rl/server/__init__.py
COPY server/app.py /tmp/rl/server/app.py
COPY server/chess_environment.py /tmp/rl/server/chess_environment.py
COPY server/static /tmp/rl/server/static

# Move files to moonfish package location
RUN MOONFISH_PATH=$(python -c "import moonfish; import os; print(os.path.dirname(moonfish.__file__))") && \
    cp -r /tmp/rl/* ${MOONFISH_PATH}/rl/ && \
    rm -rf /tmp/rl

# Download cerebellum opening book from moonfish GitHub repo (~170MB)
RUN mkdir -p /app/opening_book && \
    curl -L -o /app/opening_book/cerebellum.bin \
    "https://github.com/luccabb/moonfish/raw/master/opening_book/cerebellum.bin"

# Expose port - HF Spaces expects 7860
EXPOSE 7860

# Run the server on HF's expected port
CMD ["python", "-m", "uvicorn", "moonfish.rl.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
