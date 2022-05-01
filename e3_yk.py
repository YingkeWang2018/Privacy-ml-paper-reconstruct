from unittest import result
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
import matplotlib.pyplot as plt
from tensorflow.python.keras.metrics import Metric
# from sklearn.metrics import accuracy_score
import copy
import os

import numpy as np
import dill

tf.get_logger().setLevel('ERROR')

# import tensorflow_privacy

# from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy


def add_gradient_noise(t, noise_multiplier, clip, stddev=1.0):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise."""


    # tf.random.set_seed(1234)
    for i in range(len(t)):
        noise = tf.random.normal(shape=tf.shape(t[i]), mean=0.0, stddev=stddev * noise_multiplier * clip, dtype=tf.float32)
        t[i] += noise
    return t


def add_gradient_noise_yk(t, noise_multiplier, clip, stddev=1.0):
    """
    t: compressed gradient using yk's method
    noise_multiplier:
    clip: l2_norm_clip
    """
    cpy_t = copy.deepcopy(t)
    noised_t = add_gradient_noise(t, noise_multiplier, clip, stddev)
    for i in range(len(noised_t)):
        np_arr = noised_t[i].numpy()
        np_cp = cpy_t[i].numpy()
        np_arr[np_cp == 0.0] = 0.0
        noised_t[i] = tf.convert_to_tensor(np_arr)
    return noised_t



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

def perform_top_k_jimmy(grad, percentile):
    population = np.abs(np.concatenate([np.reshape(w0, (-1,)) for w0 in grad]))
    population_size = len(population)
    k = int(population_size * percentile)
    k = 1
    ind = np.argpartition(population, -k)[-k:]
    threshold = np.min(population[ind])
    if percentile == 0 or k == 0:
        threshold = np.inf

    for i in range(len(grad)):
        np_arr = grad[i].numpy()
        np_arr[(np_arr < threshold) & (np_arr > -threshold)] = 0.0
        grad[i] = tf.convert_to_tensor(np_arr)
    return grad


def perform_percentile_compression(grad_org, nc_grad, p_ind, percentile):
    """
    grad_org: raw gradient, not compressed
    nc_grad: processed gradient, noised, clipped, not compressed
    p_ind: percentile of indices to preserve from grad_org
    return: compressed nc_grad with p_ind*percentile% indices kept from grad_org and the rest be the top values
            from the leftover of nc_grad
    """
    population = np.abs(np.concatenate([np.reshape(w0, (-1,)) for w0 in grad_org]))
    population_size = len(population)


    # since we are trying to get the indices of p_ind% of original gradient, so k here is original k * p_ind
    k = int(population_size * percentile * p_ind)
    ind = np.argpartition(population, -k)[-k:]
    org_threshold = np.min(population[ind])
    # if we want to keep no info from raw gradient, then set boundary to include everything
    if k == 0:
        org_threshold = np.inf

    cp_nc_grad = copy.deepcopy(nc_grad)
    # first round those value we want to preserve to 0.0 first, then choose the threshold from the rest non-0.0 value
    for i in range(len(nc_grad)):
        np_nc_grad = nc_grad[i].numpy()
        np_nc_grad[(grad_org[i].numpy() >= org_threshold) | (grad_org[i].numpy() <= -org_threshold)] = 0.0
        nc_grad[i] = tf.convert_to_tensor(np_nc_grad)
    population_nc = np.abs(np.concatenate([np.reshape(w0, (-1,)) for w0 in nc_grad]))
    # this variable means the number of gradient values we should choose from nc_grad
    leftover_k = int(int(population_size * percentile) - k)
    lo_ind = np.argpartition(population_nc, -leftover_k)[-leftover_k:]
    lo_threshold = np.min(population_nc[lo_ind])

    for i in range(len(cp_nc_grad)):
        arr = cp_nc_grad[i].numpy()
        arr[(grad_org[i].numpy() > -org_threshold) & (grad_org[i].numpy() < org_threshold) & (
                    cp_nc_grad[i].numpy() > -lo_threshold) & (cp_nc_grad[i].numpy() < lo_threshold)] = 0.0
        cp_nc_grad[i] = tf.convert_to_tensor(arr)
    return cp_nc_grad


def perform_coin_flip(cell_val, toss_p):
    # perform flip
    if toss_p <= 0.01:
        if cell_val != 0:
            cell_val = 0
        else:
            cell_val = 1
    return cell_val


def perform_jim_alg(grad, grad_org, p_ind, percentile):
    grad_org_preserve = perform_top_k_jimmy(grad_org, percentile)
    # add coin flip method
    for i in range(len(grad_org_preserve)):
        org_preser_np = grad_org_preserve[i].numpy()
        for idx, x in np.ndenumerate(org_preser_np):
            random_n = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float64)
            org_preser_np[idx] = perform_coin_flip(x, random_n)
        grad_org_preserve[i] = tf.convert_to_tensor(org_preser_np)
    # perform new top-k on noise grad:
    for i in range(len(grad)):
        org_preser_np = grad_org[i].numpy()
        noise_np = grad[i].numpy()
        for idx, x in np.ndenumerate(org_preser_np):
            if x == 0:
                noise_np[idx] = 0
        grad[i] = tf.convert_to_tensor(noise_np)
    population = np.abs(np.concatenate([np.reshape(w0, (-1,)) for w0 in grad]))
    print('each iteration percentage', np.count_nonzero(population)/len(population))

    return grad


def perform_top_k_noise_top_k(grad, percentile, noise_multiplier, clip):
    # grad = perform_top_k(grad, percentile)
    grad = add_gradient_noise(grad, noise_multiplier, clip)
    grad = perform_top_k(grad, percentile)
    return grad


def perform_yk_compression_2(grad, grad_org, p_ind, percentile):
    result = copy.deepcopy(grad)
    grad_org = perform_top_k(grad_org, percentile)
    grad_org_preserve = copy.deepcopy(grad_org)
    num_preserved = 0
    # add coin flip method
    for idx, x in np.ndenumerate(grad_org_preserve):
        random_n = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float64)
        grad_org_preserve[idx] = perform_coin_flip(x, random_n)
    for i in range(len(grad_org)):
        org_np = grad_org[i].numpy()
        noise_np = grad[i].numpy()
        org_preser_np = grad_org_preserve[i].numpy()
        nz_ind = np.nonzero(org_np)
        nz_ind = np.asarray(nz_ind)
        nnz_shape = nz_ind.shape
        for i_non_zero in range(nnz_shape[1]):
            random_n = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float64)
            a = tuple(nz_ind[:, i_non_zero])
            if random_n < 0.5:
                noise_np[a] = 0
                num_preserved += 1
            else:
                org_preser_np[tuple(nz_ind[:, i_non_zero])] = 0
        # only the preversed index get saved
        grad_org_preserve[i] = tf.convert_to_tensor(org_preser_np)
        grad[i] = tf.convert_to_tensor(noise_np)
    # the the population after zeroing all preverved index
    population = np.abs(np.concatenate([np.reshape(w0, (-1,)) for w0 in grad]))
    leftover_k = int(len(population) * percentile) - num_preserved
    ind = np.argpartition(population, -leftover_k)[-leftover_k:]
    threshold = np.min(population[ind])
    for i in range(len(grad)):
        np_arr = grad[i].numpy()
        np_arr[(np_arr < threshold) & (np_arr > -threshold)] = 0.0

        org_preser_np = grad_org_preserve[i].numpy()
        nz_ind = np.nonzero(org_preser_np)
        nz_ind = np.asarray(nz_ind)
        nnz_shape = nz_ind.shape
        for i_non_zero in range(nnz_shape[1]):
            # assign the presevred noisy index back
            np_arr[tuple(nz_ind[:, i_non_zero])] = result[i][tuple(nz_ind[:, i_non_zero])]
        grad[i] = tf.convert_to_tensor(np_arr)
    return grad





def perform_yk_compression(grad, grad_org, p_ind, percentile):
    """
    grad: clipped & noised gradient
    grad_org: raw gradient
    p_ind: probability of keeping each index
    percentile: compression rate
    """
    result = copy.deepcopy(grad)
    grad_org = perform_top_k(grad_org, percentile)


    for i in range(len(grad_org)):
        org_np = grad_org[i].numpy()
        nz_ind_foo = np.nonzero(org_np)
        non_zero_value = org_np[nz_ind_foo]
        nz_ind = np.transpose(np.nonzero(org_np))
        for ind in nz_ind:
            random_n = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float64)
            if random_n > p_ind:
                org_np[ind[0]] = 0.0
        grad_org[i] = tf.convert_to_tensor(org_np)

    # count number of non-zero elements chosen
    population = np.abs(np.concatenate([np.reshape(w0, (-1,)) for w0 in grad_org]))
    count_non_zero = np.count_nonzero(population)

    leftover_k = int(len(population) * percentile) - count_non_zero

    for i in range(len(grad)):
        np_grad = grad[i].numpy()
        np_org = grad_org[i].numpy()
        np_grad[np_org != 0.0] = 0.0
        grad[i] = tf.convert_to_tensor(np_grad)

    grad_popu = np.abs(np.concatenate([np.reshape(w0, (-1,)) for w0 in grad]))
    ind = np.argpartition(grad_popu, -leftover_k)[-leftover_k:]
    threshold = np.min(grad_popu[ind])

    for i in range(len(result)):
        np_res = result[i].numpy()
        np_res[(grad_org[i].numpy() == 0.0) & ((grad[i].numpy() > -threshold) & (grad[i].numpy() < threshold))] = 0.0
        result[i] = tf.convert_to_tensor(np_res)

    return result


def perform_T(grad, power):
    for i in range(len(grad)):
        np_arr = grad[i].numpy()
        np_arr = np.power(np_arr, power)
        grad[i] = tf.convert_to_tensor(np_arr)
    return grad



def undo_T(grad, power):
    for i in range(len(grad)):
        np_arr = grad[i].numpy()
        np_arr = np.cbrt(np_arr)
        # np_arr = np.power(np_arr, 1 / power)
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


if __name__ == '__main__':
    tf.random.set_seed(1234)
    # Use CPU instead of GPU, FUCK Apple M1
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

    train_labels = np.array(train_labels, dtype=np.int32)  # [0, 4, 7, 8]

    test_labels = np.array(test_labels, dtype=np.int32)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)  # convert into one-hot format

    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    assert train_data.min() == 0.
    assert train_data.max() == 1.
    assert test_data.min() == 0.
    assert test_data.max() == 1.

    epochs = 300
    l2_norm_clip = 1.3
    std_dev = 1.0
    learning_rate = 0.01
    noise_multiplier = 0.1  # 0.01
    percentile = 0.01
    ind_percentile = 0.8

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

    # print(model.summary())

    cross_ent = tf.compat.v1.losses.softmax_cross_entropy
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    loss_arr = []
    acc_arr = []
    indices_loss = []

    for _ in range(epochs):
        with tf.GradientTape() as tape:
            y = model(train_data)

            loss = cross_ent(train_labels, y)
            loss_arr.append(loss.numpy())

        model.compile(optimizer=optimizer, loss=cross_ent, metrics=['accuracy'])
        results = model.evaluate(test_data, test_labels)
        acc_arr.append(results[1])

        # WHERE we have gradient
        grad = tape.gradient(loss, model.trainable_variables)
        grad_org = copy.deepcopy(grad)

        grad, _ = tf.clip_by_global_norm(grad, l2_norm_clip)

        noised_grad = add_gradient_noise(grad, noise_multiplier, l2_norm_clip, std_dev)

        # noised_grad = add_gradient_noise_yk(grad, noise_multiplier, l2_norm_clip, std_dev)

        # grad_com = perform_jim_alg(noised_grad, grad_org, ind_percentile, percentile)  # WORKS
        grad_com = perform_top_k_noise_top_k(grad_org, percentile, noise_multiplier, l2_norm_clip)

        optimizer.apply_gradients(zip(grad_com, model.trainable_variables))

    with open("n=0.1,c=1.3,ind=normal_acc", "wb") as dill_file:
        dill.dump(acc_arr, dill_file)
