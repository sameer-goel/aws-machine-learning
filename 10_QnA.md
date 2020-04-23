## The Dockerfile

We then create a Dockerfile with our dependencies and define the program that will be executed in SageMaker.

```
FROM tensorflow/tensorflow:2.0.0a0

RUN pip install sagemaker-containers

# Copies the training code inside the container
COPY train.py /opt/ml/code/train.py

# Defines train.py as script entry point
ENV **SAGEMAKER_PROGRAM** train.py

```

Ref: https://github.com/aws/sagemaker-containers/blob/master/README.rst