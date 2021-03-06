from unittest import result
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.metrics import Metric
import copy
import os
from scipy.spatial import distance
import numpy as np
import dill
from numpy.linalg import svd
import time
NUM_BUCKET = 10
NUM_MEAN_ITER = 30


def add_gradient_noise(t, noise_multiplier, clip, stddev=1.0, name="add_gradient_noise"):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise."""

    t = t[0]
    for i in range(len(t)):
        noise = tf.random.normal(shape=tf.shape(t[i]), mean=0.0, stddev=stddev * noise_multiplier * clip,
                                 dtype=tf.float32)
        t[i] += noise
    return t


def perform_top_k(grad, percentile):
    population = np.abs(np.concatenate([np.reshape(w0, (-1,)) for w0 in grad]))
    population_size = len(population)
    k = int(population_size * percentile)
    ind = np.argpartition(population, -k)[-k:]
    threshold = np.min(population[ind])
    if percentile == 0 or k == 0:
        threshold = np.inf

    for i in range(len(grad)):
        np_arr = grad[i].numpy()
        np_arr[(np_arr < threshold) & (np_arr > -threshold)] = 0.0
        grad[i] = tf.convert_to_tensor(np_arr)
    return grad


def eval_top_k_indices(grad, nc_grad, percentile):
    grad = perform_top_k(grad, percentile)
    # nc_grad = perform_top_k(nc_grad, percentile)
    population = np.abs(np.concatenate([np.reshape(w0, (-1,)) for w0 in grad]))
    population_size = len(population)

    grad = np.concatenate([np.reshape(w0, (-1,)) for w0 in grad])
    nc_grad = np.concatenate([np.reshape(w0, (-1,)) for w0 in nc_grad])

    zero_grad = np.count_nonzero(grad == 0.0)

    grad[nc_grad == 0.0] = 0

    new_zero_grad = np.count_nonzero(grad == 0.0)
    return (new_zero_grad - zero_grad) / (population_size * percentile)


def convert_gradients_np(grad_lst):
    result = []
    for grad in grad_lst:
        grad_np = []
        for i in range(len(grad)):
            np_arr = list(grad[i])
            grad_np.append(np_arr)
        result.append(grad_np)
    return result


def random_sample_buckets(gradient_lst, num_bucket):
    """split the gradients into sub buckets"""
    result_lst = [[] for _ in range(num_bucket)]
    for gradient in gradient_lst:
        bucket_indx = np.random.randint(num_bucket)
        result_lst[bucket_indx].append(gradient)
    return result_lst


def calculate_median_of_means(gradient_lst):
    """get the initial estimate using median of means """
    means = []
    for gradient_sub in gradient_lst:
        means.append(np.mean(gradient_sub, axis=0))

    return means, np.median(means, axis=0)


def reshape_grad_lst(gradient_lst):
    """reshape the gradient_lst to return a list of samples of each individual gradient"""
    gradients_in_model = len(gradient_lst[0])
    gradients_lst = [[] for _ in range(gradients_in_model)]
    for sample_gradient in gradient_lst:
        for i in range(len(sample_gradient)):
            gradients_lst[i].append(sample_gradient[i])
    return gradients_lst

def paper_mean_alg(gradient_lst):
    sample_gradients_lst = reshape_grad_lst(gradient_lst)
    papaer_mean = []
    for gradient_lst in sample_gradients_lst:
        bucket_gradients = random_sample_buckets(gradient_lst, NUM_BUCKET)
        bucket_means, x_0 = calculate_median_of_means(bucket_gradients)
        bucket_means = pruning_scaling(bucket_means, x_0)
        # get optimal theta
        lower_theta = 0
        higher_theta = 1
        # approxBregman(bucket_means, (higher_theta + lower_theta)/2, NUM_MEAN_ITER)
        papaer_mean.append(x_0)
    return papaer_mean



def pruning_scaling(bucket_means, x_0):
    """preprocessing step"""
    # pruning part
    dsts =[]
    for bucket_mean in bucket_means:
        dsts.append((bucket_mean, distance.euclidean(bucket_mean.flatten(), x_0.flatten())))
    dsts.sort(reverse=True, key=lambda tup: tup[1])
    dsts_reduce = dsts[round(len(bucket_means)/10):]
    max_edu = dsts_reduce[0][1]
    dsts_reduce = [item[0] for item in dsts_reduce]

    return np.array([1/max_edu *(dst - x_0) for dst in dsts_reduce])

def get_optimal_theta():
    """Distance estimate function"""

    pass


def approxBregman(bucket_means, theta, num_iterations):
    """approxBergman algorithm"""
    initial_weight = np.ones(len(bucket_means)) * 1/len(bucket_means)
    for t in range(num_iterations):
        A_t = []
        for i in range(bucket_means):
            A_t.append(np.sqrt(initial_weight[i]) * bucket_means[i])
            A_t = np.array(A_t)
            U,S,VT = svd(A_t, full_matrices=False)
            w_t = VT[0]
            w_t = w_t.reshape((-1, 1))
            sigma = np.abs(bucket_means @ w_t)





if __name__ == '__main__':
    seed = 1234
    tf.random.set_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # load build in dataset provided by keras
    train, test = tf.keras.datasets.mnist.load_data()

    train_data, train_labels = train
    test_data, test_labels = test

    # to cut the training time, do a slicing on the training data, there are 60,000 sample points before
    train_data = train_data[0: 5000]
    train_labels = train_labels[0: 5000]
    test_data = test_data[0: 1000]
    test_labels = test_labels[0: 1000]

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    # print(train_data.shape)
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
    # print(test_data.shape)

    train_labels = np.array(train_labels, dtype=np.int32) # [0, 4, 7, 8]
    
    test_labels = np.array(test_labels, dtype=np.int32)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10) # convert into one-hot format
    
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    assert train_data.min() == 0.
    assert train_data.max() == 1.
    assert test_data.min() == 0.
    assert test_data.max() == 1.

    epochs = 40
    l2_norm_clip = 2.3
    std_dev = 1.0
    learning_rate = 0.01
    noise_multiplier = 0.03 # 0.01
    percentile = 0.01
    num_microbatch = 500

    assert len(train_data) % num_microbatch == 0
    # assert epochs * (5000 / num_microbatch) == 1000 # Only want 1000 updates, 5000 is training data length

    thres1 = 1
    thres2 = 5
    ao1 = 0.9

    # choose the LeNet model 
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 8,
                            strides=2,
                            padding='same',
                            activation='relu',
                            input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Conv2D(32, 4,
                            strides=2,
                            padding='valid',
                            activation='relu'),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

   
    cross_ent = tf.compat.v1.losses.softmax_cross_entropy
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    loss_arr = []
    acc_arr = []
    indices_loss = []

    for _ in range(epochs):
        for _ in range(int(len(train_data) / num_microbatch)):
            model.compile(optimizer=optimizer, loss=cross_ent, metrics=['accuracy'])
            results = model.evaluate(test_data, test_labels)
            acc_arr.append(results[1])

            grad_arr = []
            # random generate indices 
            indices = np.random.randint(0, len(train_data), size=num_microbatch)
            for i in range(len(indices)):
                with tf.GradientTape(persistent=True) as tape:
                    idx = indices[i]
                    y = model(train_data[idx].reshape(1, 28, 28, 1))
                    loss = cross_ent(train_labels[idx].reshape(1, 10), y)
                grad_arr.append(tape.gradient(loss, model.trainable_variables))

            del tape

            for i in range(len(grad_arr)):
                grad_arr[i] = tf.clip_by_global_norm(grad_arr[i], l2_norm_clip)
                grad_arr[i] = add_gradient_noise(grad_arr[i], noise_multiplier, l2_norm_clip, std_dev)

            # standard mean
            start_time = time.time()
            mean_ = np.mean(grad_arr, axis=0)
            # mean_ = paper_mean_alg(grad_arr)
            print(f'meam Alg duration {time.time() - start_time}')
            # selected_grad = []
            # selected_grad = np.mean(grad_arr, axis=0)
            grad_com = perform_top_k(mean_, percentile)
            optimizer.apply_gradients(zip(grad_com, model.trainable_variables))
        

    filename = "p=" + str(percentile) + ",n=" + str(noise_multiplier) + ",c=" + str(l2_norm_clip) + "_seed=" + str(seed) + "_batch=" + str(num_microbatch) + "paper_acc"

    with open(filename, "wb") as dill_file:
        dill.dump(acc_arr, dill_file)
    