# FROM tiangolo/uvicorn-gunicorn-fastapi
# COPY . /usr/app/
# EXPOSE 8000
# WORKDIR /usr/app/
# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN pip install -r requirements.txt
# CMD python main.py

#FROM continuumio/anaconda3
FROM python
WORKDIR /usr/app/
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY . .
EXPOSE $PORT
#CMD uvicorn main:app --workers=4 --host 0.0.0.0 --port $PORT 
CMD ["uvicorn", "main:app", "--host=0.0.0.0"]


