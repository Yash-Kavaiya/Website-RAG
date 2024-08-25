FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
COPY . .
RUN pip install -r requirements.txt
WORKDIR /app
EXPOSE 8080
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENTRYPOINT ["python", "main.py"]