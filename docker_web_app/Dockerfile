FROM python:3.9
COPY ./requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./app /app
WORKDIR /app
CMD [ "flask", "--app", "app", "run", "--host=0.0.0.0"]