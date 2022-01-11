# Power Consumption Prediction Model
(January 2022) All credit to
Eric Lefort
Xavier Cossettini 
Mehul Bhardwaj

## Overview

Using __TensorFlow__ primarily, we have developed a neural network deep learning model that predicts power consumption in a city based on various factors such as time of day, date, temperature, wind speed, etc.

Our model was constructed around the following dataset: https://archive.ics.uci.edu/ml/datasets/Power+consumption+of+Tetouan+city

This was our team's submission to the Daisy Intelligence 2022 hackathon.

## Neural Network

We used a simple neural network architecture, consisting of a sequential network with 3 dense/fully-connected layers and 2 dropout layers. Our model achieved a prediction accuracy of 89% and a score of 0.74 on the R-squared metric. (averaged over performance in the three distinct zones)