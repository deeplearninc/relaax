# RL Simple Model Server 

[DataFlow & Class Diagram](https://goo.gl/photos/bzn1wQ28kct8vJgC6)

Befor you start, make sure you have installed on you system:

- python 2.7
- pip
- virtualenvs (http://docs.python-guide.org/en/latest/dev/virtualenvs/#virtualenvironments-ref)

To get started run the following commands:

- mkvirtualenv simple_model_server
- pip install -r requirements.txt
- gunicorn -k flask_sockets.worker server:app
- navigate to localhost:8000 in the browser

