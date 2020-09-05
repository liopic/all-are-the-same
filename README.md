# All Are The Same

A small project to encode Spanish politicians' pictures into 2 dimensions and build a map.

## Objectives

The objectives of this project are:

### 1. Political/philosophic

Show that all congress' members look alike, despite their affiliation.

### 2. Tech

Explore the use of **different virtualization solutions** in Python, that could take advantage of a native GPU despite the virtualization layer. A priori, one option is ```pipenv```, to control both dependencies and environment. Another option is docker.

Also **use ```Keras``` as interface** for creating neural networks. The idea is training a autoencoder model, storing it and applying it later to create a visual result.

## Use

### With pipenv

Use pipenv to install and open a shell. You can use it too to check code style with ```pipenv run style```. Unluckily this **will not use your GPU**.

### With docker

If you have a machine with a NVIDIA GPU, **using a docker container is the only virtualization option that will use the GPU**.

You need to have both docker and NVIDIA drivers installed first. Then install [nvidia for docker](https://github.com/NVIDIA/nvidia-docker/blob/master/README.md#quickstart) in your machine, and you will be able to use any tensorflow docker container. Look for the correct tag, like version-gpu, for instance 2.1.0-gpu.

There is a Dockerfile ready to work with. Just create the container with:
```
docker build -t all-are-the-same .
```

Later use it to execute anything, like:
```
docker run -t -v $(pwd):/app all-are-the-same python 2.train_autoencoder.py
```

## Lessons learned
- *Keras* : Creating an autoencoder is extremely simple with Keras.
- *Keras* : Using MSE makes decoded images too blurry, so it'd better to use other error scoring (like perceptuals).
- *pipenv* : Some of the Cuda libraries need by tensorflow to use a GPU are not possible to install in a virtual environment (virtualenv or pipenv), so in order to use a GPU in a virtualized environment, docker is the only easy solution.
- *docker* : The only option to virtualize a project that uses a local GPU.

## ToDo
- [X] Update to tensorflow 2
- [X] Explain some lessons learned
- [X] Use docker
- [X] Complete the code for the 2-dim map
- [ ] Try other loss functions, like perceptual ones
