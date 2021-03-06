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

import itertools
import time

import numpy as np
from pyspark import SparkContext, SparkConf, SparkFiles
import scipy.sparse as sparse

from scipy.sparse.linalg import svds
from scipy.sparse.linalg import spsolve

LAMBDA = 0.8   # regularization
# np.random.seed(42)
NUM_USER = 100000
NUM_SONG = 1000
NUM_ITER = 5
NUM_PARTITION = 5
K = 40

# dirty global
num_zeros = None
alpha = None
total = None

def fill_maxtrix(line, counts):
    global num_zeros
    global total

    # print line
    line = line.split('\t')
    user = int(line[0]) - 1
    item = int(line[1]) - 1
    count = float(line[2])

    if count > 0:
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
        sc,
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

    url = "s3n://spark-mllib/fastcode/data/" + filename
    # url = "hdfs://localhost:9000/data/" + filename
    print 'loading... ' + url
    # data = sc.textFile(url)
    # data.map(lambda l: fill_maxtrix(l, counts))

    sc.addFile(url)
    with open(SparkFiles.get(filename)) as f:
        for line in f:
            fill_maxtrix(line, counts)

    alpha = num_zeros / total
    print 'alpha %.2f' % alpha
    counts *= alpha

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
        # print user_vectors[row, :]
        # print item_vectors[col, :]
        predict = user_vectors[row, :].dot(item_vectors[col, :].T)
        if count > 0:
            err += ((1 + count) * (predict - 1) ** 2)
        else:
            err += ((1 + count) * (predict - 0) ** 2)
        numerator += 1
    if numerator == 0:
        return 0
    else:
        return err / numerator


def update(i, vec, fixed_vecs, R, YtY, user, eye, lambda_eye):
    if user:
        num_fixed = NUM_USER
        counts_i = R[:, i].T.toarray()
    else:
        num_fixed = NUM_SONG
        counts_i = R[i, :].toarray()

    num_factors = K
    # counts_i = R[i, :].toarray()
    CuI = sparse.diags(counts_i, [0])
    pu = counts_i.copy()
    pu[np.where(pu != 0)] = 1.0
    
    YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
    YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)
    
    return spsolve(YtY + YTCuIY + lambda_eye, YTCupu)

if __name__ == "__main__":

    """
    Usage: als [M] [U] [F] [iterations] [partitions]"
    """
    appName = "PythonALS"
    conf = SparkConf().setAppName(appName)
    sc = SparkContext(conf=conf)

    M = NUM_SONG
    U = NUM_USER
    F = K
    ITERATIONS = NUM_ITER
    partitions = NUM_PARTITION

    print "Running ALS with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n" % (
        M, U, F, ITERATIONS, partitions
    )

    R, nonzero = load_matrix("sorted_train_data.txt", sc)
    R, validates = partition_train_data(
        R,
        nonzero,
    )
    us, ms = svd(R)
    
    us_arr = us
    ms_arr = ms
    
    train_error = evaluate_error(R, us_arr, ms_arr)
    print '[SVD] Training Error:', train_error, "\n"

    us = sparse.csr_matrix(us_arr)
    ms = sparse.csr_matrix(ms_arr)

    print "Start broadcast"
    Rb = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)
    eye_ub = sc.broadcast(sparse.eye(NUM_USER))
    eye_mb = sc.broadcast(sparse.eye(NUM_SONG))
    lambda_eyeb = sc.broadcast(LAMBDA * sparse.eye(K))

    print "End broadcast"

    total_0 = time.time()

    for i in range(ITERATIONS):
        print "ITERATIONS:", i
        t0 = time.time()
        print "Start update ms"
        XtX = sparse.csr_matrix(us_arr.T.dot(us_arr))
        XtXb = sc.broadcast(XtX)
        ms = sc.parallelize(range(M), partitions) \
               .map(lambda x: update(
                    x, msb.value[x, :], 
                    usb.value, Rb.value, 
                    XtXb.value, True, 
                    eye_ub.value, lambda_eyeb.value)) \
               .collect()
        
        ms_arr = np.array(ms)
        ms = sparse.csr_matrix(ms_arr)
        msb = sc.broadcast(ms)

        print "Start update us"
        YtY = sparse.csr_matrix(ms_arr.T.dot(ms_arr))
        YtYb = sc.broadcast(YtY)
        us = sc.parallelize(range(U), partitions) \
               .map(lambda x: update(
                    x, usb.value[x, :], 
                    msb.value, Rb.value, 
                    YtYb.value, False, 
                    eye_mb.value, lambda_eyeb.value)) \
               .collect()
        
        us_arr = np.array(us)
        us = sparse.csr_matrix(us_arr)
        usb = sc.broadcast(us)

        print "Start evaluating:"

        train_error = evaluate_error(R, us_arr, ms_arr)
        validate_error = evaluate_error(validates, us_arr, ms_arr)
        print "Iteration %d:" % i
        print "\TrainERR: %5.4f, \ValidateERR: %5.4f\n" % (
            train_error, validate_error)
        t1 = time.time()
        print 'Finished in %f seconds\n' % (t1 - t0)

    total_1 = time.time()
    print 'ALS Finished in %f seconds\n' % (total_1 - total_0)
    sc.stop()
