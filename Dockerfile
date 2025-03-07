FROM python:3.13

WORKDIR /app

# Install uv
RUN pip install uv

RUN uv init 
# Install fireducks and other dependencies using uv
RUN uv add fireducks streamlit pandas plotly pydeck requests numpy

# Expose port for Streamlit
EXPOSE 8501

# No ENTRYPOINT - we'll use docker exec to run commands directly

# Default command (just keeps container running)
CMD ["tail", "-f", "/dev/null"]
