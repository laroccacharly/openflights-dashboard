FROM python:3.13

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install uv
RUN pip install uv

RUN uv init 
# Install fireducks and other dependencies using uv
RUN uv add fireducks streamlit pandas plotly pydeck requests numpy

# Expose port for Streamlit
EXPOSE 8501

COPY main.py .
COPY src/ src/

CMD ["uv", "run", "streamlit", "run", "--server.address=0.0.0.0", "--server.port=8501", "main.py"]
