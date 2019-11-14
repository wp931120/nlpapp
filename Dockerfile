FROM python:3.6.4

RUN mkdir /app

ADD . /app
WORKDIR /app
RUN pip install git+https://www.github.com/bojone/bert4keras.git 
RUN pip install -r requirements.txt
EXPOSE 8890
CMD ["python", "app.py"]
