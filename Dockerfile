FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install system dependencies if needed
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project configuration
COPY app/pyproject.toml ./

# Copy the main algorithm module
COPY app/graph.py ./
COPY app/__init__.py ./

# Install the package in development mode
RUN pip install -e .[dev]

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password="]