# TF_Tools

## Description

This repository contains a collection of tools for TensorFlow. The tools are organized in subfolders. Each subfolder contains a README.md file with a description of the tool and a link to the source code.

## Contents

### Quantization

It contains tools for quantizing a TensorFlow model.

* h5_tflite.py: It converts a Keras model in HDF5 format to a TensorFlow Lite model.
* jax2tflite.py: It converts a JAX model to a TensorFlow model.

### Flops Counter

It contains tools for counting the number of floating point operations (FLOPs) of a TensorFlow model.

* get_flops.py: It counts the number of FLOPs of a TensorFlow model.
