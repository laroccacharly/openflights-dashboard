FROM python:3.13

WORKDIR /app

# Install uv
RUN pip install uv

RUN uv init 
# Install fireducks using uv
RUN uv add fireducks

# No ENTRYPOINT - we'll use docker exec to run commands directly

# Default command (just keeps container running)
CMD ["tail", "-f", "/dev/null"]
