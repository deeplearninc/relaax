# RL Simple Model Server 

Before you start, make sure you have installed on your system:

- `python 2.7 or 3.5`

- [`pip`](https://pip.pypa.io/en/stable/installing/) - just need to install requirements, see command below:
    > pip install -r requirements.txt

To get started run the following commands:

    > gunicorn -k flask_sockets.worker server:app

If installation of tensorflow failed - perhaps you are on `Linux` and have choice.
Delete tensorflow note from `requirements.txt` and install [Tensorflow manually](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html).

#### [How to create a virtual environment](/VirtualEnvironments.md)

[DataFlow & Class Diagram](https://goo.gl/photos/bzn1wQ28kct8vJgC6)

