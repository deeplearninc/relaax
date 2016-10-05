# How to create a virtual environment

A Virtual Environment is a tool to keep the dependencies required
by different projects in separate places, by creating virtual Python
environments for them. It solves the “Project X depends on version
1.x, but Project Y needs 4.x” dilemma, and keeps your global
site-packages directory clean and manageable.

## Virtualenv

[virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/#virtualenvironments-ref) is a tool to create isolated Python environments.
virtualenv creates a folder which contains all the necessary
executables to use the packages that a Python project would need.

> You need to install python package manager before ([pip](https://pip.pypa.io/en/stable/installing/) for example)

To get started run the following commands (`Linux`):

- Go to your project root or another desired folder, for example:

    `cd PROJECT_DIR`

- Then you should create your environment there with pointing to your
python interpreter link and type your desired environment name:

    `virtualenv -p /usr/bin/python2.7 ENVIRONMENT_NAME`

Your environment is ready to use from now!

- To activate your environment to working on just type this:

    `source ENVIRONMENT_NAME/bin/activate`
    
- If you finished your work you can deactivate your environment by:

    `deactivate`

## Anaconda

In progress... 

(I prefer Anaconda for more complex and high-load tasks)