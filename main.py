import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import nltk
import string
from nltk.corpus import stopwords

print("Using TF version", tf.__version__)

import pdb

STOP_WORDS = set(stopwords.words('english'))
EMBEDDING_DIM = 100 # Glove embedding size

contradictions = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}


def cleanString(sentence):
    return sentence


def replace_contradictions(word):
    if word in contradictions:
        return contradictions[word]
    else:
        return word

def clean_news(sentence):
    new_sentence = []
    tokens = nltk.word_tokenize(sentence)

    # Lowercase everything
    tokens = [w.lower() for w in tokens]

    words = [replace_contradictions(w) for w in tokens]

    # Remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # Remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # Remove stop words
    words = [w for w in words if not w in STOP_WORDS]

    # Remove words with less than 1 character
    words = [w for w in words if len(w) > 1]

    return words



def loadDataCombinedColumns(DATA_URL):
    data = pd.read_csv(
        DATA_URL, parse_dates=True, index_col=0, verbose=True, keep_default_na=False
    )
    data["combined_news"] = data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)
    data["combined_news"] = data["combined_news"].apply(clean_news)
    data_y = data["Label"]
    # data_X = data.iloc[:, 1:26]
    data_X = data["combined_news"]
    # data_X = data.iloc[:,1:26].apply(lambda headline:cleanString(' '.join(headline)), axis=1)

    test_X = data_X["2015-01-02":"2016-07-01"]
    train_X = data_X["2008-08-08":"2014-12-31"]
    test_y = data_y["2015-01-02":"2016-07-01"]
    train_y = data_y["2008-08-08":"2014-12-31"]
    return (train_X, train_y, test_X, test_y)


def mergeNews(x_df, y_df):
    new_x = []
    new_y = []
    # for date_idx in range(y_df.size):
        # y_label = y_df.iloc[date_idx]
        # news = x_df.iloc[date_idx]
        # for news_idx in news:
            # new_x.extend(news.values)
            # new_y.extend([y_label] * len(news.values))
    for date_idx in range(y_df.size):
        y_label = y_df.iloc[date_idx]
        x_news = x_df.iloc[date_idx]

        new_x.append(x_news)
        new_y.append(y_label)

    return new_x, new_y

def encode_and_pad_data(data, tokenizer, max_length):
    # integer encode the text data
    encoded_data = tokenizer.texts_to_sequences(data)

    # maps index to word
    word_index = tokenizer.word_index

    # pad the vectors to create uniform length
    padded_data = keras.preprocessing.sequence.pad_sequences(
        encoded_data, maxlen=max_length, padding="post"
    )
    return padded_data


def create_glove_embeddings():

    # Load the GLOVE embeddings
    embeddings_index = {}
    with open("glove/glove.6B.100d.txt", encoding="utf-8") as f:
        for line in f:
            values = line.split(" ")
            word = values[0]
            embedding = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = embedding

    print("Word embeddings:", len(embeddings_index))
    return embeddings_index


def basic_model(embedding_layer, vocab_size):
    model = tf.keras.Sequential()
    model.add(embedding_layer)
    model.add(tf.keras.layers.GlobalMaxPool1D())
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return model

def lstm_model(embedding_layer, vocab_size):
    model = tf.keras.Sequential()
    hidden_size=50

    model.add(embedding_layer)
    # model.add(tf.keras.layers.LSTM(hidden_size, return_sequences=True))
    # model.add(tf.keras.layers.Dropout(0.2))

    # model.add(tf.keras.layers.LSTM(hidden_size, return_sequences=True))
    # model.add(tf.keras.layers.Dropout(0.2))

    # model.add(tf.keras.layers.LSTM(hidden_size, return_sequences=True))
    # model.add(tf.keras.layers.Dropout(0.2))

    # model.add(tf.keras.layers.LSTM(hidden_size, return_sequences=True))
    # model.add(tf.keras.layers.Dropout(0.2))

    # model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # model.add(Embedding(n_vocab,100))
    # model.add(tf.keras.layers.Dropout(0.25))
    # model.add(tf.keras.layers.SimpleRNN(100,return_sequences=True))
    # model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid')))
    model.add(tf.keras.layers.LSTM(50, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(tf.keras.layers.LSTM(50, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model


def main():

    train_x, train_y, test_x, test_y = loadDataCombinedColumns("./stocknews/Combined_News_DJIA.csv")
    cleaned_train_x, cleaned_train_y = mergeNews(train_x, train_y)
    cleaned_test_x, cleaned_test_y = mergeNews(test_x, test_y)

    print("Length of data")
    print(len(cleaned_train_x), len(cleaned_train_y))


    # get the longest new article length
    max_length = len(max(cleaned_train_x, key=len))
    print("Max length", max_length)
    print("Number of news training examples", len(cleaned_train_x))


    # initialize the tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer()

    tokenizer.fit_on_texts(cleaned_train_x)
    vocab_size = len(tokenizer.word_index) + 1

    train_x_encoded = encode_and_pad_data(cleaned_train_x, tokenizer, max_length)
    test_x_encoded = encode_and_pad_data(cleaned_test_x, tokenizer, max_length)

    embeddings_index = create_glove_embeddings()

    # Create the embedding matrix
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    print("non_zero_elements", nonzero_elements / vocab_size)

    embedding_layer = tf.keras.layers.Embedding(
        len(tokenizer.word_index) + 1,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=False,
    )

    # Create the model
    # model = basic_model(embedding_layer)
    model = lstm_model(embedding_layer, vocab_size)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    train_x_np = np.array(train_x_encoded)
    train_y_np = np.array(cleaned_train_y)

    test_x_np = np.array(test_x_encoded)
    test_y_np = np.array(cleaned_test_y)

    print("Train_x_shape", train_x_np.shape)
    print("Train_y_shape", train_y_np.shape)

    print(test_y_np)

    history = model.fit(
        train_x_np,
        train_y_np,
        epochs=30,
        verbose=1,
        validation_data=(test_x_np, test_y_np),
        batch_size=32,
    )
    loss, accuracy = model.evaluate(train_x_np, train_y_np, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(test_x_np, test_y_np, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))



if __name__ == "__main__":
    main()
