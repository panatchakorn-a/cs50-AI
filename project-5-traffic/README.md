## Experimentation process
I firstly tried with a model with one convolutional layer, max pooling layer, hidden layer, and dropout layer, and the model didn't learn.
When I changed dropout rate from 0.5 to 0.2, the accuracy disruptly improved accuracy from 5% to 86%, suggesting that the dropout rate value is very significant. Also, adding dropout layer after each convolutional and hidden layer helps preventing overfitting (high accuracy in training but low in testing).
Then, adding more convolutional layer and filter sizes also makes the model improved to over 90% accuracy. However, three-layers case was worse which I think it's because at that point the input size is already too small for another convolution.
Adding up pooling size to over 2 didn't help improving accuracy in my case.
Finally, in my case, adding up size of the hidden layer helps while adding more hidden layers doesn't improve the performance of the model.