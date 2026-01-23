# digit-recognizer
A simple neural network classifier to identify draw digits from 0 to 9, trained on the MNIST dataset. I made this for fun and educational purposes.

Made using only numpy and pygame. The network itself consists of two hidden layers, with 256 and 64 hidden units respectively. The training algorithm uses mini-batch gradient descent enabling the model to achieve 98.13% accuracy on the MNIST testing data.

The canvas in draw.py is a 280 by 280 pixel grid, the input to which is processed, scaled-down and normalized to make it similar to the MNIST training data and suitable for the model to make a prediction on. You can show the predictions panel by pressing space, and clear the canvas by pressing C.
