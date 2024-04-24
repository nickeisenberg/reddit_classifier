FROM python:3.11-slim

WORKDIR /flask_app

ENV PYTHONPATH /flask_app

COPY ./app /flask_app/app
COPY ./src /flask_app/src

COPY rec_app.txt /flask_app/rec_app.txt

RUN pip install --no-cache-dir -r rec_app.txt

# get cpu pytorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


EXPOSE 5000

CMD ["python", "app"]
