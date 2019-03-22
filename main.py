import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
import re
import time
from string import punctuation

from tensorflow.keras.callbacks import TensorBoard

print("Using TF version", tf.__version__)

STOP_WORDS = set(stopwords.words("english"))
TOP_WORDS = 10000
EMBEDDING_DIM = 100  # Glove embedding size
MAX_LENGTH = 400


def clean_news(text):
    # tokens = nltk.word_tokenize(sentence)

    # # Lowercase everything
    # tokens = [w.lower() for w in tokens]

    # words = [replace_contradictions(w) for w in tokens]

    # # Remove punctuation from each word
    # table = str.maketrans('', '', string.punctuation)
    # stripped = [w.translate(table) for w in tokens]

    # # Remove remaining tokens that are not alphabetic
    # words = [word for word in stripped if word.isalpha()]

    # # Remove stop words
    # words = [w for w in words if not w in STOP_WORDS]

    # # Remove words with less than 1 character
    # words = [w for w in words if len(w) > 1]

    # return words
    cleaned_text = text.replace("\n", "").replace('"b', "").replace("'b", "")
    for punc in list(punctuation):
        cleaned_text = cleaned_text.replace(punc, "").lower()
    cleaned_text = re.sub(" +", " ", cleaned_text)
    return cleaned_text


def loadDataCombinedColumns(DATA_URL):
    data = pd.read_csv(
        DATA_URL, parse_dates=True, index_col=0, verbose=True, keep_default_na=False
    )
    data["combined_news"] = data.filter(regex=("Top.*")).apply(
        lambda x: "".join(str(x.values)), axis=1
    )
    data["combined_news"] = data["combined_news"].apply(clean_news)
    data_y = data["Label"]
    data_x = data["combined_news"]
    test_x = data_x["2015-01-02":"2016-07-01"]
    test_y = data_y["2015-01-02":"2016-07-01"]
    train_x = data_x["2008-08-08":"2014-12-31"]
    train_y = data_y["2008-08-08":"2014-12-31"]

    return (train_x, train_y, test_x, test_y)


def mergeNews(x_df, y_df):
    new_x = []
    new_y = []
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


def lstm_model_2_layers(embedding_layer, vocab_size):
    model = tf.keras.Sequential()

    model.add(embedding_layer)
    model.add(
        tf.keras.layers.LSTM(
            128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True
        )
    )
    model.add(tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model


def lstm_model_1_layers(embedding_layer, vocab_size):
    model = tf.keras.Sequential()

    model.add(embedding_layer)
    model.add(tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model


def main():
    train_x, train_y, test_x, test_y = loadDataCombinedColumns(
        "./stocknews/Combined_News_DJIA.csv"
    )
    cleaned_train_x, cleaned_train_y = mergeNews(train_x, train_y)
    cleaned_test_x, cleaned_test_y = mergeNews(test_x, test_y)

    print("Length of data", len(cleaned_train_x), len(cleaned_train_y))

    # get the longest new article length
    # max_length = len(max(cleaned_train_x, key=len))
    # print("Max length", max_length)
    # print("Number of news training examples", len(cleaned_train_x))

    # initialize the tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=TOP_WORDS)

    tokenizer.fit_on_texts(cleaned_train_x)
    vocab_size = len(tokenizer.word_index) + 1

    train_x_encoded = encode_and_pad_data(cleaned_train_x, tokenizer, MAX_LENGTH)
    test_x_encoded = encode_and_pad_data(cleaned_test_x, tokenizer, MAX_LENGTH)

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
        input_length=MAX_LENGTH,
        trainable=False,
    )

    # Create the model
    # model = basic_model(embedding_layer, vocab_size)
    # model = lstm_model_2_layers(embedding_layer, vocab_size)
    model = lstm_model_1_layers(embedding_layer, vocab_size)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    train_x_np = np.array(train_x_encoded)
    train_y_np = np.array(cleaned_train_y)

    test_x_np = np.array(test_x_encoded)
    test_y_np = np.array(cleaned_test_y)

    print("Train_x_shape", train_x_np.shape)
    print("Train_y_shape", train_y_np.shape)

    tensorboard = TensorBoard(
        log_dir="./tensorboard_logs",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
    )

    history = model.fit(
        train_x_np,
        train_y_np,
        epochs=30,
        verbose=1,
        validation_data=(test_x_np, test_y_np),
        batch_size=32,
        callbacks=[tensorboard],
    )
    loss, accuracy = model.evaluate(train_x_np, train_y_np, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(test_x_np, test_y_np, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))


if __name__ == "__main__":
    main()
