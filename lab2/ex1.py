import numpy as np
import matplotlib.pyplot as plt

coeff = [29, 44]
mius = [0.1 * x for x in range(0, 5)]
sigmas = np.sqrt([0.1 * x for x in range(1, 5)])
delta = 5
n_samples = 420


def generate_x(line, miu, sigma1, sigma2, d, n):
    x = np.random.normal(miu, sigma1 * d, n)
    epsilon = np.random.normal(miu, sigma2 * d, n)
    y = line[0] * x + line[1] + epsilon
    return x, y


for miu in mius:
    plt.figure(figsize=(16, 4))
    for j, sigma in enumerate(sigmas):
        epsilon = np.random.normal(miu, sigma, n_samples)
        x = np.random.normal(miu, sigma, n_samples)
        y = coeff[0] * x + coeff[1] + epsilon

        regular_x, regular_y = x, y
        highx_x, highx_y = generate_x(coeff, miu, sigma * delta, sigma, 1, 10)
        highy_x, highy_y = generate_x(coeff, miu, sigma, sigma * delta, 1, 12)
        highxy_x, highxy_y = generate_x(coeff, miu, sigma * delta, sigma * delta, 1, 9)

        x_total = np.concatenate([regular_x, highx_x, highy_x, highxy_x])
        y_total = np.concatenate([regular_y, highx_y, highy_y, highxy_y])

        X = np.column_stack([x_total, np.ones(len(x_total))])
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        leverage_scores = np.diag(H)
        top = np.argsort(leverage_scores)[-10:]

        plt.subplot(1, len(sigmas), j + 1)
        plt.scatter(regular_x, regular_y, c='blue', s=10)
        plt.scatter(highx_x, highx_y, c='orange', s=10)
        plt.scatter(highy_x, highy_y, c='green', s=10)
        plt.scatter(highxy_x, highxy_y, c='red', s=10)
        plt.scatter(x_total[top], y_total[top], facecolors='none', edgecolors='k', s=100)

        xl = np.linspace(x_total.min(), x_total.max(), 100)      # the line
        plt.plot(xl, coeff[0] * xl + coeff[1], 'k--')

        plt.title(f'miu={miu:.1f}, sigma^2={sigma**2:.1f}')


coeff2 = [29, 18, 44]


def generate_x_2(line, miu, sigma1, sigma2, d, n):
    x1 = np.random.normal(miu, sigma1 * d, n)
    x2 = np.random.normal(miu, sigma1 * d, n)
    epsilon = np.random.normal(miu, sigma2 * d, n)
    y = line[0] * x1 + line[1] * x2 + line[2] + epsilon
    return x1, x2, y


for miu in mius:
    plt.figure(figsize=(16, 4))
    for j, sigma in enumerate(sigmas):
        x1 = np.random.normal(miu, sigma, n_samples)
        x2 = np.random.normal(miu, sigma, n_samples)
        epsilon = np.random.normal(miu, sigma, n_samples)
        y = coeff2[0] * x1 + coeff2[1] * x2 + coeff2[2] + epsilon

        regular_x, regular_y = x, y
        highx_x, highx_y = generate_x(coeff, miu, sigma * delta, sigma, 1, 10)
        highy_x, highy_y = generate_x(coeff, miu, sigma, sigma * delta, 1, 12)
        highxy_x, highxy_y = generate_x(coeff, miu, sigma * delta, sigma * delta, 1, 9)

        x_total = np.concatenate([regular_x, highx_x, highy_x, highxy_x])
        y_total = np.concatenate([regular_y, highx_y, highy_y, highxy_y])

        X = np.column_stack([x_total, np.ones(len(x_total))])
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        leverage_scores = np.diag(H)
        top = np.argsort(leverage_scores)[-10:]

        plt.subplot(1, len(sigmas), j + 1)
        plt.scatter(regular_x, regular_y, c='blue', s=10)
        plt.scatter(highx_x, highx_y, c='orange', s=10)
        plt.scatter(highy_x, highy_y, c='green', s=10)
        plt.scatter(highxy_x, highxy_y, c='red', s=10)
        plt.scatter(x_total[top], y_total[top], facecolors='none', edgecolors='k', s=100)

        xl = np.linspace(x_total.min(), x_total.max(), 100)      # the line
        plt.plot(xl, coeff[0] * xl + coeff[1], 'k--')

        plt.title(f'miu={miu:.1f}, sigma^2={sigma**2:.1f}')

plt.show()


#de adaugat o legenda ffs.
