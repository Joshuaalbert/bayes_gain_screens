from .update import Update
import tensorflow.compat.v1 as tf
from .. import float_type
from ..misc import diagonal_jitter
from scipy.special import hermite, jacobi
import numpy as np


class UpdateLDS(Update):
    """
    Calculates the update equation assuming the observable is a linear
    transforming of the state (Y(t)=C.X(t)). C must be specified.
    """

    def __init__(self, C, **kwargs):
        super(UpdateLDS, self).__init__(**kwargs)
        self.C = C

    def _forward(self, samples, *serve_values):
        """
        Computes the data-domain samples by pushing forward.
        :param samples: tf.Tensor
            [S, B, K]
        :return: tf.Tensor
            [S, B, N]
        """
        #[N, K].[S, B, K]->[S,B,N]
        return tf.einsum("nk,sbk->sbn",tf.convert_to_tensor(self.C, dtype=float_type, name='C'), samples)

    def __call__(self, t, prior_mu, prior_Gamma, y, Sigma, *served_values):
        """
        If p(X) = N[prior_mu, prior_Gamma]
        then this computes,

        p(X | y, Sigma) = p(y | X, Sigma) p(X) / p(y | Sigma)

        :param prior_mu: tf.Tensor
            [K]
        :param prior_Gamma: tf.Tensor
            [K,K]
        :param y: tf.Tensor
            [N]
        :param Sigma: tf.Tensor
            [N,N]
        :return:
            if p(X | y, Sigma) is a Gaussian then this returns:
            the mean [K] and the covariance [K, K] of type tf.Tensor
        """

        # [N, K]
        C = tf.convert_to_tensor(self.C, dtype=float_type, name='C')

        N = tf.shape(C)[0]
        K = tf.shape(C)[1]

        ###
        # get posterior
        # N, K
        CGamma = tf.matmul(C, prior_Gamma)
        # N, N
        CGammaCT = tf.matmul(CGamma, C, transpose_b=True)
        # N, K
        KT = tf.linalg.lstsq(Sigma + CGammaCT + diagonal_jitter(N), CGamma, fast=False)
        # [N,K].[K]->[N]
        Cmu = tf.linalg.matmul(C, prior_mu[:, None])[:, 0]
        # [N,K].[N,1]->[K]
        post_mu = prior_mu + tf.linalg.matmul(KT, (y - Cmu)[:, None], transpose_a=True)[:, 0]
        # [N,K].[ N,K]->[ K, K]
        KC = tf.linalg.matmul(KT, C, transpose_a=True)
        # K,K
        post_Gamma = tf.linalg.matmul(tf.eye(K, dtype=float_type) - KC, prior_Gamma)

        return post_mu, post_Gamma


class UpdateLDSPolynomial(UpdateLDS):
    def __init__(self, x, deg=2, basis='jacobi', scale = False,  **kwargs):

        K = deg + 1
        if scale:
            x = (x - x.min())
            x = x / x.max()
        if basis == 'jacobi':
            # N, K
            X = np.stack([jacobi(k, 0., 0.)(x) for k in range(K)], axis=1)
        elif basis == 'hermite':
            # N, K
            X = np.stack([hermite(k)(x) for k in range(K)], axis=1)
        else:
            # N, K
            X = np.stack([x ** k for k in range(K)], axis=1)
        if scale:
            X /= np.sqrt(np.sum(X ** 2, axis=0))
        super(UpdateLDSPolynomial, self).__init__(X, **kwargs)
