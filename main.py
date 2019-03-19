import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np

print("Using TF version", tf.__version__)

import pdb


def cleanString(sentence):
    return sentence


def loadDataCombinedColumns(DATA_URL):
    data = pd.read_csv(
        DATA_URL, parse_dates=True, index_col=0, verbose=True, keep_default_na=False
    )
    data["combined_news"] = data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)
    data_y = data["Label"]
    data_X = data.iloc[:, 1:26]
        # data_X = data.iloc[:,1:26].apply(lambda headline:cleanString('BLAH '.join(headline)), axis=1)

    test_X = data_X["2015-01-02":"2016-07-01"]
    train_X = data_X["2008-08-08":"2014-12-31"]
    test_y = data_y["2015-01-02":"2016-07-01"]
    train_y = data_y["2008-08-08":"2014-12-31"]
    return (train_X, train_y, test_X, test_y)


def mergeNews(x_df, y_df):
    new_x = []
    new_y = []
    for date_idx in range(y_df.size):
        y_label = y_df.iloc[date_idx]
        news = x_df.iloc[date_idx]
        for news_idx in news:
            new_x.extend(news.values)
            new_y.extend([y_label] * len(news.values))
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




def main():
    EMBEDDING_DIM = 100

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

    e = tf.keras.layers.Embedding(
        len(tokenizer.word_index) + 1,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=False,
    )

    model = tf.keras.Sequential()
    model.add(e)
    model.add(tf.keras.layers.GlobalMaxPool1D())
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()


    # print(train_x_encoded.shape)
    # print(np.arraycleaned_train_y))


    train_x_np = np.array(train_x_encoded)
    train_y_np = np.array(cleaned_train_y)

    test_x_np = np.array(test_x_encoded)
    test_y_np = np.array(cleaned_test_y)

    print(train_x_np.shape)
    print(train_y_np.shape)

    # print(train_x_encoded[0])
    history = model.fit(
        train_x_np,
        train_y_np,
        epochs=50,
        verbose=1,
        validation_data=(test_x_np, test_y_np),
        batch_size=10,
    )
    # loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    # print("Training Accuracy: {:.4f}".format(accuracy))
    # loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    # print("Testing Accuracy:  {:.4f}".format(accuracy))



if __name__ == "__main__":
    main()
