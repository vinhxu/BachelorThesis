# imports
import tensorflow as tf
import matplotlib.pyplot as plt
from load_data import *
import time

startTime = time.time()



# Load data
x_train, y_train, x_valid, y_valid = load_data(mode="train")
x_test, y_test = load_data(mode='test')

n_inputs = 2
n_classes = 2

# Hyper-parameters
epochs = 20000         # Total number of training epochs
batch_size = 3000      # Training batch size
display_freq = 100     # Frequency of displaying the training results
learning_rate = 0.001  # initial learning rate




def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


# weight and bais wrappers
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    # initer = tf.truncated_normal_initializer(stddev=0.0001)
    initer = tf.random_normal_initializer()
    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)

def fc_layer(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    in_dim = x.get_shape()[1]
    W = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])
    layer = tf.matmul(x, W)
    layer += b
    if use_relu:
        layer = tf.nn.relu(layer)
    else:
        layer = tf.nn.sigmoid(layer)
    return layer

h1 = 8   # Number of units in the first hidden layer
h2 = 6   # Number of units in the second hidden layer
h3 = 4   # Number of units in the  hidden layer

# Create the graph for the linear model
# Placeholders for inputs (x) and outputs(y)
x = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')


fc1 = fc_layer(x, h1, 'FC1', use_relu=True)
fc2 = fc_layer(fc1, h2, 'FC2', use_relu=True)
fc3 = fc_layer(fc2, h3, 'FC3', use_relu=True)
output_logits = fc_layer(fc3, n_classes, 'OUT', use_relu=True)
# print ('output_logits: ' + str(output_logits))

# Network predictions
cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')
y_true_test  = tf.argmax(y_test, axis=1, name='y_true_test')
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')


# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)



# Create the op for initializing all variables
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)

saver = tf.train.Saver()

def train_nn_model(x_train, y_train, saveModelBoolean = True):
    global_step = 0
    # Number of training iterations in each epoch
    num_tr_iter = int(len(y_train) / batch_size)
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch + 1))
        x_train, y_train = randomize(x_train, y_train)
        for iteration in range(num_tr_iter):
            global_step += 1
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(x_train, y_train, start, end)

            # Run optimization op (backprop)
            feed_dict_batch = {x: x_batch, y: y_batch}
            sess.run(optimizer, feed_dict=feed_dict_batch)

            if iteration % display_freq == 0:
                # Calculate and display the batch loss and accuracy
                loss_batch, acc_batch = sess.run([loss, accuracy],
                                                 feed_dict=feed_dict_batch)

                print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                      format(iteration, loss_batch, acc_batch))

        # Run validation after every epoch
        feed_dict_valid = {x: x_valid[:], y: y_valid[:]}
        loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
              format(epoch + 1, loss_valid, acc_valid))
        print('---------------------------------------------------------')

        # if (epoch>2000):
        #     endTime = time.time()
        #     print(endTime - startTime)
        #     break

    if (saveModelBoolean):
        modelName = "/Users/vinhxu/Desktop/gitBT/03_savedModel/.route1_10k_5.2.5_i864o_Loss{0:.2f}_Accuracy{1:.01%}.ckpt".format(loss_valid, acc_valid)
        save_path1 = saver.save(sess, modelName)
        latestModelName = "/Users/vinhxu/Desktop/gitBT/03_savedModel/5.2.5_i864o_.ckpt".format(loss_valid, acc_valid)
        save_path2 = saver.save(sess, latestModelName)


def plot_images(images, cls_true, cls_pred=None, title=None, plot=plt, color='b'):

    ax = plt.gca()
    ax.plot(images[:,0], images[:,1], marker="o", color=color, ls="", ms=.7)


def plot_example_errors(images, cls_true, cls_pred, title=None):

    # Negate the boolean array.
    incorrect = np.logical_not(np.equal(cls_pred, cls_true))

    # Get the images from the test-set that have been
    # incorrectly classified.
    incorrect_images = images[incorrect]

    # Get the true and predicted classes for those images.
    cls_pred = cls_pred[incorrect]
    cls_true = cls_true[incorrect]

    # Plot the images.
    plot_images(images=incorrect_images[:],
                cls_true=cls_true[:],
                cls_pred=cls_pred[:],
                title=title,color='red')


def plot_example_correct(images, cls_true, cls_pred, title=None):

    # Negate the boolean array.
    correct = np.equal(cls_pred, cls_true).astype(bool)

    # Get the images from the test-set that have been
    # incorrectly classified.
    correct_images = images[correct]

    # Get the true and predicted classes for those images.
    cls_pred = cls_pred[correct]
    cls_true = cls_true[correct]

    # Plot the images.
    plot_images(images=correct_images[:],
                cls_true=cls_true[:],
                cls_pred=cls_pred[:],
                title=title,color='g')


def get_test_date(dataName="test"):
    if (dataName == "train"):
        return {x: x_train[:], y: y_train[:]}
    elif (dataName == "valid"):
        return {x: x_valid[:], y: y_valid[:]}
    else:
        return {x: x_test[:], y: y_test[:]}

def test_nn_model(modelName="/Users/vinhxu/Desktop/gitBT/03_savedModel/.1nn_validation_Loss0.13_Accuracy97.1%.ckpt", dataName = "test", loadModelBoolean = True):
    if (loadModelBoolean):
        # Restore variables from disk.
        saver.restore(sess, modelName)

    # Get test data
    feed_dict_test = get_test_date(dataName)
    print feed_dict_test

    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
    print('---------------------------------------------------------')
    print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
    print('---------------------------------------------------------')

    # Plot some of the correct and misclassified examples
    plt.figure()
    output = sess.run(output_logits, feed_dict=feed_dict_test)
    cls_pred = sess.run(cls_prediction, feed_dict=feed_dict_test)
    cls_true = np.argmax(feed_dict_test.get(y), axis=1)
    plot_images(feed_dict_test.get(x), cls_true, cls_pred, title='Correct Examples', color='green')
    plot_example_errors(feed_dict_test.get(x), cls_true, cls_pred, title='Misclassified Examples')
    plt.show()

    for i, c in enumerate(cls_true):
        print "output: " + str(output[i]) + "  , cls_pred: " + str(cls_pred[i]) + "  , correct: " + str(c)
    print(len(cls_true))


walkingPath = {x: walkingPathData[:,1:], y: convert_to_oneHot(walkingPathData[:,0])}

def routeCheck(modelName="./03_savedModel/.Latest_route1.ckpt", dataName = walkingPath, loadModelBoolean = True):

    if (loadModelBoolean):
        # Restore variables from disk.
        saver.restore(sess, modelName)

    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=walkingPath)
    print('---------------------------------------------------------')
    print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
    print('---------------------------------------------------------')

    # Plot some of the correct and misclassified examples
    plt.figure()
    output = sess.run(output_logits, feed_dict=walkingPath)
    cls_pred = sess.run(cls_prediction, feed_dict=walkingPath)
    cls_true = np.argmax(walkingPath.get(y), axis=1)
    plot_images(walkingPath.get(x), cls_true, cls_pred, title='Correct Examples', color='green')
    plot_example_errors(walkingPath.get(x), cls_true, cls_pred, title='Misclassified Examples')
    plt.show()

    for i, c in enumerate(cls_true):
        print "output: " + str(output[i]) + "  , cls_pred: " + str(cls_pred[i]) + "  , correct: " + str(c)


# routeCheck('./03_savedModel/5.2.5_i8o_.ckpt')
# routeCheck('./03_savedModel/.1nn_validation_Loss0.13_Accuracy97.1%.ckpt')
train_nn_model(x_train, y_train, saveModelBoolean=True)
# test_nn_model(loadModelBoolean=False)
# test_nn_model('./03_savedModel/5.2.5_i444o_.ckpt', dataName='train')
# test_nn_model('./03_savedModel/5.2.5_i444o_.ckpt', dataName='valid')
# test_nn_model('./03_savedModel/5.2.5_i444o_.ckpt', dataName='test')

endTime = time.time()

print(endTime-startTime)

# >>> import datetime
# >>> str(datetime.timedelta(seconds=666))
# '0:11:06'