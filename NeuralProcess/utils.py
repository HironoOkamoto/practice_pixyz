import numpy as np

def context_target_random_split(train_X, train_y):
    N = train_X.shape[0]
    perm = np.random.permutation(np.arange(N))
    train_X, train_y = train_X[perm], train_y[perm]

    N_c = np.random.choice(np.arange(1, N))
    x_c, y_c = train_X[:N_c], train_y[:N_c]
    x_t, y_t = train_X[N_c:], train_y[N_c:]
    return x_c, x_t, y_c, y_t