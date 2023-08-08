FROM python:3.9.12-slim
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN apt-get update
RUN apt-get install default-jdk -y
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . .
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
