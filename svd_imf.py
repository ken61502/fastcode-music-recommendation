import time
import itertools
import numpy as np
import scipy.sparse as sparse

from scipy.sparse.linalg import svds
from scipy.sparse.linalg import spsolve

NUM_USER = 100000
NUM_SONG = 1000
NUM_ITER = 1
alpha = None
K = 40


def load_matrix(filename, num_users=NUM_USER, num_items=NUM_SONG):
    global alpha

    print "Start to load matrix...\n"
    t0 = time.time()
    counts = np.zeros((num_users, num_items))
    total = 0.0
    num_zeros = num_users * num_items
    for i, line in enumerate(open(filename, 'r')):
        user, item, count = line.strip().split('\t')
        user = int(user) - 1
        item = int(item) - 1
        count = float(count)
        if user > num_users:
            continue
        if item > num_items:
            continue
        if count != 0:
            counts[user, item] = count
            total += count
            num_zeros -= 1
        if i % 100000 == 0:
            print 'loaded %i counts...' % i
    alpha = num_zeros / total
    print 'alpha %.2f' % alpha
    counts *= alpha
    counts = sparse.csr_matrix(counts)
    t1 = time.time()
    print 'Finished loading matrix in %f seconds\n' % (t1 - t0)
    print 'Total entry:', num_users * num_items
    print 'Non-zeros:', num_users * num_items - num_zeros
    return counts, num_users * num_items - num_zeros


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
            err += ((1 + alpha * count) * (predict - 1) ** 2)
        else:
            err += ((1 + alpha * count) * (predict - 0) ** 2)
        numerator += 1
    if numerator == 0:
        return 0
    else:
        return err / numerator


class ImplicitMF():

    def __init__(
        self,
        counts,
        user_vectors=None,
        item_vectors=None,
        num_factors=K,
        num_iterations=NUM_ITER,
        reg_param=0.8
    ):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param
        self.user_vectors = user_vectors
        self.item_vectors = item_vectors

    def train_model(self):
        # Initialize X, Y matrix randomly if no Q, P is passed
        if self.user_vectors is None or self.item_vectors is None:
            self.user_vectors = np.random.normal(size=(self.num_users,
                                                       self.num_factors))
            self.item_vectors = np.random.normal(size=(self.num_items,
                                                       self.num_factors))

        for i in xrange(self.num_iterations):
            t0 = time.time()
            print 'Solving for user vectors...'
            self.user_vectors = self.iteration(
                True, sparse.csr_matrix(self.item_vectors)
            )
            print 'Solving for item vectors...'
            self.item_vectors = self.iteration(
                False, sparse.csr_matrix(self.user_vectors)
            )
            t1 = time.time()
            print 'iteration %i finished in %f seconds' % (i + 1, t1 - t0)

    def iteration(self, user, fixed_vecs):
        num_solve = self.num_users if user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)
        eye = sparse.eye(num_fixed)
        lambda_eye = self.reg_param * sparse.eye(self.num_factors)
        solve_vecs = np.zeros((num_solve, self.num_factors))

        t = time.time()
        for i in xrange(num_solve):
            if user:
                counts_i = self.counts[i].toarray()
            else:
                counts_i = self.counts[:, i].T.toarray()
            CuI = sparse.diags(counts_i, [0])
            pu = counts_i.copy()
            pu[np.where(pu != 0)] = 1.0
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            solve_vecs[i] = xu
            if i % 1000 == 0:
                print 'Solved %i vecs in %d seconds' % (i, time.time() - t)
                t = time.time()

        return solve_vecs


if __name__ == '__main__':
    counts, nonzero = load_matrix(
        '../../data/sorted_train_data.txt',
    )
    counts, validates = partition_train_data(
        counts,
        nonzero,
    )

    user_vectors, item_vectors = svd(counts)

    train_err = evaluate_error(counts, user_vectors, item_vectors)
    print 'Training Error:', train_err

    mf = ImplicitMF(counts, user_vectors, item_vectors)
    mf.train_model()

    # Evaluate training and validation error
    train_err = evaluate_error(counts, mf.user_vectors, mf.item_vectors)
    val_err = evaluate_error(validates, mf.user_vectors, mf.item_vectors)
    print 'Training Error:', train_err
    print 'Validation Error:', val_err
