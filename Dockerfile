FROM python:3.11-slim

WORKDIR /app

COPY ./app /app

COPY rec_app.txt /app/rec_app.txt
RUN pip install --no-cache-dir -r rec_app.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

EXPOSE 5000

CMD ["python", "__main__.py"]
