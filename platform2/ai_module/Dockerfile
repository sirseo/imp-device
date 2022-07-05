FROM python:3.8-buster

WORKDIR /ai_module

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD [ "python", "./main.py" ]
#CMD [ "python", "./predict.py" ]

EXPOSE 80
