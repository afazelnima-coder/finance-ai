FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY agents/         agents/
COPY utils/          utils/
COPY rag/            rag/
COPY streamlit_app.py .
COPY main.py          .
COPY mcp_server.py    .
COPY mcp_http_server.py .

# Default command: run the Streamlit web app.
# Use CMD (not ENTRYPOINT) so docker-compose can override per service.
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
