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
from __future__ import print_function

import sys
import itertools

import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark import SparkContext
import scipy.sparse as sparse

from scipy.sparse.linalg import svds
from scipy.sparse.linalg import spsolve

LAMBDA = 0.01   # regularization
np.random.seed(42)


def rmse(R, ms, us):
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / M * U)


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

    print("""WARN: This is a naive implementation of ALS and is given as an
      example. Please use the ALS method found in pyspark.mllib.recommendation
      for more conventional use.""", file=sys.stderr)

    sc = SparkContext(appName="PythonALS")
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    U = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    F = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    ITERATIONS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    partitions = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    print("Running ALS with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n" %
          (M, U, F, ITERATIONS, partitions))


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
        validate_error = evaluate_error(validate, ms, us)
        print("Iteration %d:" % i)
        print("\TrainERR: %5.4f, \ValidateERR: %5.4f\n" % (
            train_error, validate_error)
        )

    sc.stop()
