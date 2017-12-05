# Tensorflow Example Scripts

These scripts work through the set of TensorFlow Tutorials [here](https://www.tensorflow.org/get_started/)

They have been designed to allow individuals with some exposure to Python/ML to explore the tensorflow library.

Current Files:

- intro: a set of scripts exploring the basic building blocks of tensorflow
- MNIST: three sequential models that explore the Hello, World! of Machine Learning: the MNIST dataset.

## Local Install Instructions

Use of tensorflow requires:

- python
- python-pip
- python-dev
- python-numpy
- tensorflow

Follow the install instructions [here](https://www.tensorflow.org/install/)

## Docker Install Instructions

A Docker file is provided for those who want to run these scripts in an isolated environment:

```
docker build . -t tensorflow:latest
docker run -it tensorflow:latest bash
```

Once inside the container, scripts can be run with `python <path_to_script>` as above.