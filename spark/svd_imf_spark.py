#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This is an example implementation of ALS for learning how to use Spark.
Please refer to ALS in pyspark.mllib.recommendation for more conventional use.
This example requires numpy (http://www.numpy.org/)
"""
# from __future__ import print_function

import sys
import itertools
import time

import numpy as np
from pyspark import SparkContext
import scipy.sparse as sparse

from scipy.sparse.linalg import svds
from scipy.sparse.linalg import spsolve

LAMBDA = 0.01   # regularization
np.random.seed(42)
NUM_USER = 100000
NUM_SONG = 1000
NUM_ITER = 5
NUM_PARTITION = 2
K = 40

# dirty global
num_zeros = None
alpha = None
total = None


def rmse(R, ms, us):
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / M * U)


def fill_maxtrix(line, counts):
    global num_zeros
    global total

    line = line.split('\t')
    user = int(line[0]) - 1
    item = int(line[1]) - 1
    count = float(line[2])

    if user > NUM_USER:
        continue
    if item > NUM_SONG:
        continue
    if count != 0:
        counts[user, item] = count
        total += count
        num_zeros -= 1


def partition_train_data(
    counts,
    nonzero,
    percent=0.8,
    num_users=NUM_USER,
    num_items=NUM_SONG
):
    print "Start to partition data...\n"
    t0 = time.time()
    num_train = int(np.floor(nonzero * percent))
    num_validate = int(nonzero - num_train)

    shuffle_index = range(nonzero)
    np.random.shuffle(shuffle_index)

    validate_index = shuffle_index[:num_validate]
    shuffle_index[:num_validate].sort()

    validate_counts = sparse.lil_matrix((num_users, num_items), dtype=np.int32)
    idx, curr = 0, 0
    counts = sparse.lil_matrix(counts)
    counts_coo = counts.tocoo()
    for row, col, count in itertools.izip(counts_coo.row,
                                          counts_coo.col,
                                          counts_coo.data):
        if idx < num_validate and validate_index[idx] == curr:
            validate_counts[row, col] = count
            counts[row, col] = 0
            idx += 1
        curr += 1
    t1 = time.time()
    print 'Finished partitioning data in %f seconds\n' % (t1 - t0)
    return counts.tocsr(), validate_counts.tocoo()


def load_matrix(
        filename,
        sparkContext,
        num_users=NUM_USER,
        num_items=NUM_SONG
):
    global alpha
    global total
    global num_zeros

    print 'Start to load matrix...'

    t0 = time.time()
    counts = np.zeros((num_users, num_items))
    total = 0.0
    num_zeros = num_users * num_items

    # url = "s3n://spark-mllib/fastcode/data/" + filename
    url = "../../data/" + filename
    data = sparkContext.textFile(url)

    data.map(lambda l: fill_maxtrix(l, counts))

    alpha = num_zeros / total
    print 'alpha %.2f' % alpha
    counts *= alpha
    counts = sparse.csr_matrix(counts)
    t1 = time.time()
    print 'Finished loading matrix in %f seconds\n' % (t1 - t0)
    print 'Total entry:', num_users * num_items
    print 'Non-zeros:', num_users * num_items - num_zeros

    counts = sparse.csr_matrix(counts)

    return counts, num_users * num_items - num_zeros


def svd(train_data_mat, num_users=NUM_USER, num_songs=NUM_SONG, factors=K):
    print "start computing SVD...\n"
    start = time.time()
    U, S, VT = svds(train_data_mat, factors)
    stop = time.time()
    print "SVD done!\n"
    print "Finished SVD in " + str(stop - start) + " seconds\n"
    return U, np.diag(S).dot(VT).T


def evaluate_error(counts, user_vectors, item_vectors):
    counts_coo = counts.tocoo()
    numerator = 0
    err = 0.0
    for row, col, count in itertools.izip(counts_coo.row,
                                          counts_coo.col,
                                          counts_coo.data):
        predict = user_vectors[row, :].dot(item_vectors[col, :])
        if count > 0:
            err += ((1 + count) * (predict - 1) ** 2)
        else:
            err += ((1 + count) * (predict - 0) ** 2)
        numerator += 1
    if numerator == 0:
        return 0
    else:
        return err / numerator


def update(i, vec, fixed_vecs, ratings, YTY):
    num_fixed = fixed_vecs.shape[0]
    num_factors = fixed_vecs.shape[1]

    eye = sparse.eye(num_fixed)
    lambda_eye = LAMBDA * sparse.eye(num_factors)

    counts_i = vec.toarray()
    CuI = sparse.diags(counts_i, [0])
    pu = counts_i.copy()
    pu[np.where(pu != 0)] = 1.0

    YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
    YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)
    return spsolve(YTY + YTCuIY + lambda_eye, YTCupu)


if __name__ == "__main__":

    """
    Usage: als [M] [U] [F] [iterations] [partitions]"
    """

    sc = SparkContext(appName="PythonALS")
    M = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_SONG
    U = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_USER
    F = int(sys.argv[3]) if len(sys.argv) > 3 else K
    ITERATIONS = int(sys.argv[4]) if len(sys.argv) > 4 else NUM_ITER
    partitions = int(sys.argv[5]) if len(sys.argv) > 5 else NUM_PARTITION

    print "Running ALS with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n" % (
        M, U, F, ITERATIONS, partitions)

    R, nonzero = load_matrix("sorted_train_data.txt", sc)
    R, validates = partition_train_data(
        R,
        nonzero,
    )
    us, ms = svd(R)

    Rb = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)

    for i in range(ITERATIONS):
        mat = usb.value
        XtX = mat.T.dot(mat)
        ms = sc.parallelize(range(M), partitions) \
               .map(lambda x:
                    update(x, msb.value[x, :], usb.value, Rb.value, XtX)) \
               .collect()
        # collect() returns a list, so array ends up being
        # a 3-d array, we take the first 2 dims for the matrix
        ms = np.array(ms)[:, :, 0]
        msb = sc.broadcast(ms)

        mat = msb.value
        XtX = mat.T.dot(mat)
        us = sc.parallelize(range(U), partitions) \
               .map(lambda x:
                    update(x, usb.value[x, :], msb.value, Rb.value.T, XtX)) \
               .collect()
        us = np.array(us)[:, :, 0]
        usb = sc.broadcast(us)

        train_error = evaluate_error(R, ms, us)
        validate_error = evaluate_error(validates, ms, us)
        print "Iteration %d:" % i
        print "\TrainERR: %5.4f, \ValidateERR: %5.4f\n" % (
            train_error, validate_error)

    sc.stop()
