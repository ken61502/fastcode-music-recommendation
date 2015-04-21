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
This is an example implementation of ALS for learning how to use Spark. Please refer to
ALS in pyspark.mllib.recommendation for more conventional use.
This example requires numpy (http://www.numpy.org/)
"""
from __future__ import print_function

import sys

import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark import SparkContext

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

def load_matrix(filename, sparkContext, num_users=NUM_USER, num_items=NUM_SONG):
    global alpha
    global total
    global num_zeros

    print "Start to load matrix...\n"
    t0 = time.time()
    counts = np.zeros((num_users, num_items))
    total  = 0.0
    num_zeros = num_users * num_items

    url  = "s3n://spark-mllib/fastcode/data/" + filename
    data = sparkContext.textFile(url)

    data.map(lambda l: l.split('\t')).map(fill_maxtrix(l, counts))

    alpha = num_zeros / total
    print 'alpha %.2f' % alpha
    counts *= alpha
    counts = sparse.csr_matrix(counts)
    t1 = time.time()
    print 'Finished loading matrix in %f seconds\n' % (t1 - t0)
    print 'Total entry:', num_users * num_items
    print 'Non-zeros:', num_users * num_items - num_zeros
    return counts, num_users * num_items - num_zeros    

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

def update(i, vec, mat, ratings):
    uu = mat.shape[0]
    ff = mat.shape[1]

    XtX = mat.T * mat
    Xty = mat.T * ratings[i, :].T

    for j in range(ff):
        XtX[j, j] += LAMBDA * uu

    return np.linalg.solve(XtX, Xty)


if __name__ == "__main__":

    """
    Usage: als [M] [U] [F] [iterations] [partitions]"
    """

    print("""WARN: This is a naive implementation of ALS and is given as an
      example. Please use the ALS method found in pyspark.mllib.recommendation for more
      conventional use.""", file=sys.stderr)

    sc = SparkContext(appName="PythonALS")
    M = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_SONG
    U = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_USER
    F = int(sys.argv[3]) if len(sys.argv) > 3 else K
    ITERATIONS = int(sys.argv[4]) if len(sys.argv) > 4 else NUM_ITER
    partitions = int(sys.argv[5]) if len(sys.argv) > 5 else NUM_PARTITION

    print("Running ALS with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n" %
          (M, U, F, ITERATIONS, partitions))

    R, nonzero = load_matrix("sorted_train_data.txt", sc)
    R, validates = partition_train_data(
        counts,
        nonzero,
    )
    us, ms = svd(R)

    Rb  = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)

    for i in range(ITERATIONS):
        ms = sc.parallelize(range(M), partitions) \
               .map(lambda x: update(x, msb.value[x, :], usb.value, Rb.value)) \
               .collect()
        # collect() returns a list, so array ends up being
        # a 3-d array, we take the first 2 dims for the matrix
        ms = matrix(np.array(ms)[:, :, 0])
        msb = sc.broadcast(ms)

        us = sc.parallelize(range(U), partitions) \
               .map(lambda x: update(x, usb.value[x, :], msb.value, Rb.value.T)) \
               .collect()
        us = matrix(np.array(us)[:, :, 0])
        usb = sc.broadcast(us)

        error = rmse(R, ms, us)
        print("Iteration %d:" % i)
        print("\nRMSE: %5.4f\n" % error)

    sc.stop()