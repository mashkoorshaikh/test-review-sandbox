# Pinned minimal Python image for sandboxed test execution
FROM python:3.11-slim

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install pinned testing dependencies
RUN pip install --no-cache-dir pytest==8.3.4
