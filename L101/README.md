# L101 Project: Comparison and Evaluation of Techniques for Neural Natural Language Generation from a Meaning Representation

This work was completed as an assessed project for L101

## Prerequisites

Running this code requires the following python packages:

* Tensorflow
* Keras
* numpy
* scipy
* nltk
* pandas

The code as is will require a CUDA-enabled graphics card and tensorflow-gpu, but this can be changed if CuDNNLSTM is changed to LSTM wherever it appears

## Information

The trained models that were used to obtain the results given in my report are in the folder models

The results obtained in my testing are in the folder Results

The results were evaluated using the script provided [here](https://github.com/tuetschek/e2e-metrics), which uses BLEU, METEOR, NIST, ROUGE-L and CIDEr
