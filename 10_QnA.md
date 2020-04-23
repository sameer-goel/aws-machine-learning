## ML Cheatsheets

<img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.mathworks.com%2Fhelp%2Fstats%2Fmachine-learning-in-matlab.html&psig=AOvVaw3wR5LZh7UXAJjPipF6tajB&ust=1587698338171000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCKib067L_egCFQAAAAAdAAAAABAX" height="300" />

## Question on Dockerfile

We then create a Dockerfile with our dependencies and define the program that will be executed in SageMaker.

Remember when running a program to specify the entry point for Script Mode, set the SAGEMAKER_PROGRAM environmental variable. The script must be located in the /opt/ml/code folder.

```
FROM tensorflow/tensorflow:2.0.0a0

RUN pip install sagemaker-containers

# Copies the training code inside the container
COPY train.py /opt/ml/code/train.py

# Defines train.py as script entry point
ENV SAGEMAKER_PROGRAM train.py

```

Ref: https://github.com/aws/sagemaker-containers/blob/master/README.rst

## 