FROM python:3.9-slim

WORKDIR /app

COPY merged_model-1/merged_model /app/merged_model


COPY app.py /app/app.py

COPY requirements.txt /app/requirements.txt






RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5000


CMD ["python", "app.py"]

