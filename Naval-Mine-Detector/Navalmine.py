
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


tf.disable_v2_behavior()

def read_dataset():
    df = pd.read_csv("Navalminesonar.csv")
    print("Dataset loaded successfully")
    #print("Number of coloumns:",len(df.columns))

    X = df[df.columns[0:5]].values
    y = df[df.columns[60]]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    print("Encoded Labels:", y)
    Y = one_hot_encode(y)

    print("Value of X.shape", X.shape)
    return (X, Y)

def one_hot_encode(labels):
    n_labels = len(labels)
    print(n_labels)
    n_unique_labels = len(np.unique(labels))
    print(n_unique_labels)
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))
    out_layer = tf.add(tf.matmul(layer_4, weights['out']), biases['out'])
    return out_layer

def main():
    X, Y = read_dataset()
    X, Y = shuffle(X, Y, random_state=1)

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=415)

    learning_rate = 0.3
    training_epochs = 1000
    cost_history = []
    accuracy_history = []
    n_dim = X.shape[1]
    print("Number of columns are n_dim",n_dim)
    n_class = 2
    model_path = "ModelFlow"

    n_hidden_1 = 60
    n_hidden_2 = 60
    n_hidden_3 = 60
    n_hidden_4 = 60

    x = tf.placeholder(tf.float32, [None, n_dim])
    y_ = tf.placeholder(tf.float32, [None, n_class])

    W = tf.Variable(tf.zeros([n_dim,n_class]))
    b= tf.Variable(tf.zeros([n_class]))

    weights = {
        'h1': tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_1])),
        'h2': tf.Variable(tf.random.truncated_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random.truncated_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.random.truncated_normal([n_hidden_3, n_hidden_4])),
        'out': tf.Variable(tf.random.truncated_normal([n_hidden_4, n_class]))
    }

    biases = {
        'b1': tf.Variable(tf.random.truncated_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random.truncated_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random.truncated_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random.truncated_normal([n_hidden_4])),
        'out': tf.Variable(tf.random.truncated_normal([n_class]))
    }

    y = multilayer_perceptron(x, weights, biases)

    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            sess.run(training_step, feed_dict={x: train_x, y_: train_y})
            cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
            cost_history.append(cost)

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            acc = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
            accuracy_history.append(acc)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Cost: {cost:.4f}, Train Accuracy: {acc:.4f}")

        print("Training complete.")
        save_path = saver.save(sess, model_path)
        print(f"Model saved in file: {save_path}")

        plt.plot(accuracy_history)
        plt.title("Accuracy history")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig("Accuracy history.png")
        plt.show()

        plt.plot(range(len(cost_history)), cost_history)
        plt.title("Loss calculation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("Loss calculation.png")
        plt.show()

        pred_y = sess.run(y, feed_dict={x: test_x})
        mse = tf.reduce_mean(tf.square(pred_y - test_y))
        print("Mean Squared Error: {:.4f}".format(sess.run(mse)))

        correct_prediction = tf.equal(tf.argmax(pred_y, 1), tf.argmax(test_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Test Accuracy: {:.4f}".format(sess.run(accuracy)))

if __name__ == "__main__":
    main()
