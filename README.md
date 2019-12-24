# Dog Breed Identification App
Simple end-to-end ML project to create a Keras model, convert it using Tensorflow JS and serve it in a dockerized Angular PWA.

The model is cached in the browser, so the first time it takes some time to load, but then the network traffic is minimized.

## Data
The labelled data is taken from the Kaggle competition ["Dog Breed Identification"](https://www.kaggle.com/c/dog-breed-identification). The dataset comprises 120 breeds of dogs. The goal of the competition is to create a classifier capable of determining a dog's breed from a photo.

It contains 10222 labeled images in the training set.

## Notebooks
In this folder, notebooks with python code.

<b>Keras Model Training</b>

This notebook contains the code to train an Xception model using transfer learning. The model is loaded from Keras and pre-trained on imageNet.


## Angular App
An Angular 8 PWA with Tensorflow JS that uses the converted model to make predictions.

The model is cached in the browser and predicitons are made on the client side, so <b>network traffic is minimized</b>.
