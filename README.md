# Neural Machine Translation Model for translating Azerbaijani phrases to English.
 In this project, I have discovered how to develop a neural machine translation system for translating Azerbaijani phrases to English. I use a dataset of Azerbaijani to English terms used as the basis for flashcards for language learning. The dataset is available from the ManyThings.org website, with examples drawn from the Tatoeba Project. After cleaning text data, it will be ready for modeling and defining. I have used an encoder-decoder LSTM model on this problem. 
 
In this architecture, the input sequence is encoded by a front-end model called the encoder then decoded word by word by a backend model called the decoder. The model is trained using the efficient Adam approach to stochastic gradient descent and minimizes the categorical loss function because we have framed the prediction problem as multiclass classification. 

A plot of the model is also created providing another perspective on the model configuration.

![model](https://user-images.githubusercontent.com/31247506/87252532-90a5df80-c47c-11ea-9138-6568134b43c6.png)


Next, the model is trained. Each epoch takes about 30 seconds on modern CPU hardware; no GPU is required. Then, we can repeat this for each source phrase in a dataset and compare the predicted result to the expected target phrase in English. We can print some of these comparisons to screen to get an idea of how the model performs in practice. We will also calculate the BLEU scores to get a quantitative idea of how well the model has performed. The evaluate model() function below implements this, calling the above predict sequence() function for each phrase in a provided dataset. I discovered how to develop a neural machine translation system for translating German phrases to English. Specifically, I learned:

__How to clean and prepare data ready to train a neural machine translation system.__

__How to develop an encoder-decoder model for machine translation.__

__How to use a trained model for inference on new input phrases and evaluate the model
skill.__
