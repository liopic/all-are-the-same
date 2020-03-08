# All Are The Same

A small project to encode Spanish politicians' pictures into 2 dimensions and build a map.

## Objectives

The objectives of this project are:

### 1. Political/philosophic

Show that all congress' members look alike, despite their affiliation.

### 2. Tech

Explore the use of different virtualization solutions in Python, that could take advantage of a native GPU despite the virtualization layer. An option is ```pipenv```, to control the requeriments, enviroment and tasks of a Python project. Another option is docker.

Also use ```Keras```, training a autoencoder model, storing it and applying it later to create a visual result.

## Use

### With pipenv

Use pipenv to install and open a shell. You can use it too to check code style with ```pipenv run style```.

### With docker

TBD

## Lessons learned
- *Keras* : Creating an autoencoder is extremely simple with Keras.
- *Keras* : Using MSE makes decoded images too blurry, so it's better
- *pipenv* : Some of the Cuda libraries neeed by tensorflow to use a GPU are not possible to install in a virtual environment (virtualenv or pipenv), so in order to use a GPU in a virtualized environment, docker is the only easy solution.

## ToDo
- [X] Update to tensorflow 2
- [X] Explain some lessons learned
- [ ] Complete the code for the 2-dim map
- [ ] Use docker
- [ ] Try other loss functions, like perceptual ones
- [ ] Use python logging instead of just print
