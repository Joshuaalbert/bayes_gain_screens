"""
Code for performing Bayesian iterative reweighted least-squares (B-IRLS)

see J. Albert et al. 2019b
"""

import tensorflow as tf
import tensorflow_probability as tfp

float_type = tf.float64
import numpy as np
from bayes_gain_screens.misc import diagonal_jitter
from scipy.special import hermite, jacobi


class LDSSmoother(object):
    def __init__(self, x, times, init_sigma, order=2, Nmax=10):
        """
        Perform linear dynamics smoothing.

        :param x: np.array
            [N] coordinates data are defined over with polynomial model
        :param y: np.array
            [B, N] data in B batches
        :param times: np.array or None
            [B] times of each sample, or else assumes arange(B)
        :param order: int
            Order of polynomial model
        :return: tuple of
        smoothed data, data covariance matrix, time step covariance
        """
        N = x.shape[0]
        B = times.shape[0]
        K = order + 1
        # N,K
        #         X = np.stack([x ** k for k in range(K)], axis=1)
        x = (x - x.min())
        x = x / x.max()
        X = np.stack([jacobi(k, 0., 0.)(x) for k in range(K)], axis=1)

        X /= np.sqrt(np.sum(X ** 2, axis=0))
        graph = tf.Graph()

        with graph.as_default():
            Nmax = tf.constant(Nmax, tf.int32, name='Nmax')
            #
            X = tf.constant(X, float_type, name='X')
            X_ext = tf.tile(X[None, :, :], [B, 1, 1])
            times = tf.constant(times, float_type, name='times')
            # B-1
            dt = times[1:] - times[:-1]
            init_sigma = tf.constant(init_sigma, float_type, name='init_sigma')
            # B, N
            y_pl = tf.placeholder(float_type, shape=[B, N], name='y')
            # K,K
            XTX = tf.linalg.matmul(X, X, transpose_a=True)
            # K,K
            Gamma_0 = init_sigma ** 2 * tfp.math.pinv(XTX)
            # K, B
            XTy = tf.linalg.matmul(X, y_pl, transpose_a=True, transpose_b=True)
            # [K,K].[K,B]->[B, K]
            ols_mu = tf.transpose(tf.linalg.lstsq(XTX, XTy, fast=False), (1, 0))
            mu_0 = ols_mu[0, :]
            # K, K
            Omega_0 = tfp.stats.covariance((ols_mu[1:, :] - ols_mu[:-1, :]), sample_axis=0,
                                           event_axis=1)

            # [B,K].[N,K]->[B,N]
            predictive_y_0 = tf.linalg.matmul(ols_mu, X, transpose_b=True)
            # N
            Sigma_0 = tfp.stats.covariance(predictive_y_0 - y_pl, sample_axis=0, event_axis=1)

            ###
            # Bayesian evidence

            def cond(n, *args):
                return n < Nmax

            def body(n, mu_0_n1, Gamma_0_n1, Sigma_n1, Omega_n1, post_mu_n1, post_Gamma_n1, X):
                X_ext = tf.tile(X[None, :, :], [B, 1, 1])
                prior_Gamma_, post_mu_, post_Gamma_, last_KX = self.forward_filter(y_pl, X, mu_0_n1, Gamma_0_n1,
                                                                                   Sigma_n1, Omega_n1)
                post_mu, post_Gamma, post_Gamma_inter = self.backward_filter(y_pl, X, prior_Gamma_, post_mu_,
                                                                             post_Gamma_, last_KX)
                # B, K, K
                Pt = post_Gamma + post_mu[:, :, None] * post_mu[:, None, :]
                # B, K, K
                Ptt1 = tf.concat([tf.zeros([1, K, K], dtype=float_type),
                                  post_Gamma_inter[1:, :, :] + post_mu[1:, :, None] * post_mu[:-1, None, :]],
                                 axis=0)
                res = self.parameter_estimation(y, X_ext, post_mu, Pt, Ptt1)
                print(res)
                Sigma = res['R_new1']
                Omega = res['Q_new1']
                mu_0 = res['mu_0']
                Gamma_0 = res['Gamma_0']
                post_Gamma.set_shape(post_Gamma_n1.shape)
                post_mu.set_shape(post_mu_n1.shape)
#                 X = tf.transpose(res['C_new_T'])
                with tf.control_dependencies([tf.print('n', n),
#                                               tf.print('X', X, tf.transpose(res['C_new_T'])),
                                              tf.print('A', tf.transpose(res['A_new_T'])),
#                                               tf.print('Sigma', Sigma, res['R_new2']),
                                              tf.print('Omega', Omega, res['Q_new2']),
                                             tf.print("mu_0", mu_0_n1, mu_0),
                                             tf.print('Gamma_0', Gamma_0)]):
                    return [n + 1, mu_0, Gamma_0, Sigma, Omega, post_mu, post_Gamma, X]

            [_, _, _, Sigma, Omega, post_mu, post_Gamma, X] = \
                tf.while_loop(cond,
                              body,
                              [tf.constant(0, tf.int32),
                               mu_0,
                               Gamma_0,
                               Sigma_0,
                               Omega_0,
                               tf.zeros([B, K], float_type),
                               tf.zeros([B, K, K], float_type),
                              X])

            X_ext = tf.tile(X[None, :, :], [B, 1, 1])

            # [B,K].[N,K]->[B,N]
            predictive_y = tf.linalg.matmul(post_mu, X, transpose_b=True)
            # B, N, N
            predictive_y_Cov = Sigma + tf.linalg.matmul(X_ext, tf.linalg.matmul(post_Gamma, X_ext, transpose_b=True))

            self.y_pl = y_pl
            self.sess = tf.Session(graph=graph)
            self.result = dict(
                post_mu=post_mu,
                post_Gamma=post_Gamma,
                Omega=Omega,
                Sigma=Sigma,
                predictive_y=predictive_y,
                predictive_y_Cov=predictive_y_Cov)

    def parameter_estimation(self, y, X_ext, post_mu, Pt, Ptt1):
        """
        M-step
        :param y: [B, N]
        :param post_mu: [B, K]
        :param Pt: [B, K, K]
        :param Ptt1: [B, K, K]
        :return:
        """
        B = tf.shape(y)[0]
        N = tf.shape(X_ext)[1]
        K = tf.shape(X_ext)[2]

        # [B,K].[B,N]->[B,K,N]
        xyT = post_mu[:, :, None] * y[:, None, :]
        # [K,N]
        sum_xyT_1 = tf.reduce_sum(xyT, axis=0)
        # [K,K]
        sum_Pt_1 = tf.reduce_sum(Pt, axis=0)
        # [K, N]
        C_new_T = tf.linalg.lstsq(sum_Pt_1, sum_xyT_1)
        # [B,K,N]
        C_new_T_ext = tf.tile(C_new_T[None, :, :], [B, 1, 1])

        # N, N
        R_new1 = tf.reduce_mean(y[:, :, None] * y[:, None, :] - tf.linalg.matmul(C_new_T_ext, xyT, transpose_a=True),
                                axis=0)

        # N, N
        R_new2 = tf.reduce_mean(y[:, :, None] * y[:, None, :] - 2. * tf.linalg.matmul(X_ext, xyT) + tf.matmul(X_ext,
                                                                                                              tf.linalg.matmul(
                                                                                                                  Pt,
                                                                                                                  X_ext,
                                                                                                                  transpose_b=True)), axis=0)

        sum_Ptt1_2 = tf.reduce_sum(Ptt1[1:, :, :], axis=0)
        # K, K
        sum_Pt1t_2 = tf.transpose(sum_Ptt1_2, (1, 0))
        # K, K
        sum_Pt_2 = tf.reduce_sum(Pt[1:, :, :], axis=0)
        # K, K
        L_sum_Pt_2 = tf.linalg.cholesky(sum_Pt_2 + diagonal_jitter(K))
        # (Ptt1.L^-T.L^-1)^T = L^-T.(L^-1.Ptt1^T)

        half_A = tf.linalg.triangular_solve(L_sum_Pt_2, sum_Pt1t_2)
        # K, K
        A_new_T = tf.linalg.triangular_solve(L_sum_Pt_2, half_A, lower=False)

        # K, K
        Q_new1 = (sum_Pt_2 - tf.linalg.matmul(half_A, half_A, transpose_a=True)) / tf.cast(B - 1, float_type)

        Q_new2 = sum_Pt_2 - sum_Pt1t_2 - sum_Ptt1_2 + tf.reduce_sum(Pt[:-1, :, :], axis=0)

        mu_0 = post_mu[0, :]
        Gamma_0 = Pt[0, :, :] - mu_0[:, None] * mu_0[None, :]

        return dict(C_new_T=C_new_T,
                    R_new1=R_new1, R_new2=R_new2,
                    A_new_T=A_new_T, Q_new1=Q_new1,
                    Q_new2=Q_new2, mu_0=mu_0, Gamma_0=Gamma_0)

    def backward_filter(self, y, X, prior_Gamma_, post_mu_, post_Gamma_, last_KX):
        """

        http://mlg.eng.cam.ac.uk/zoubin/course04/tr-96-2.pdf

        Perform a forward filter pass
        :param y: tf.Tensor
            [B, N]
        :param X: tf.Tensor
            [N, K]
        :param prior_Gamma_: tf.Tensor
            [B, K, K]
        :param post_mu_: tf.Tensor
            [B,K]
        :param post_Gamma_: tf.Tensor
            [B,K,K]
        :param Sigma: tf.Tensor
            [N, N]
        :param Omega: tf.Tensor
            [K,K]
        :return: tuple
            post_mu [B,K]
            post_Gamma [B, K, K]
        """

        B = tf.shape(y)[0]
        N = tf.shape(y)[1]
        K = tf.shape(X)[1]

        # B,K,K
        JT_aug = tf.concat([tf.zeros([1, K, K], dtype=float_type),
                            tf.linalg.lstsq(prior_Gamma_[1:, :, :], post_Gamma_[:-1, :, :], fast=False)],
                           axis=0)

        def _idx(nm1):
            # B = 3, n1=2 -> 0
            # B = 3, n1=0 -> B-1
            return B - nm1 - 1

        def cond(nm1, *args):
            return nm1 >= 0

        def body(nm1, post_mu_n, post_Gamma_n, post_Gamma_inter_n, post_mu_ta, post_Gamma_ta, post_Gamma_inter_ta):
            #
            n = nm1 + 1
            nm2 = nm1 - 1
            place_nm1 = _idx(nm1)

            prior_Gamma__n = prior_Gamma_[n, :, :]
            post_Gamma__nm1 = post_Gamma_[nm1, :, :]
            post_mu__nm1 = post_mu_[nm1, :]

            # K,K
            JT_nm1 = JT_aug[nm1 + 1, :, :]
            JT_nm2 = JT_aug[nm2 + 1, :, :]
            # K
            post_mu_nm1 = post_mu__nm1 + tf.linalg.matmul(JT_nm1, (post_mu_n - post_mu__nm1)[:, None],
                                                          transpose_a=True)[:, 0]
#             # K, K
#             L_dV = tf.linalg.cholesky(post_Gamma_n - prior_Gamma__n + diagonal_jitter(K))
#             # L^T.J^T
#             # K, K
#             LTJT = tf.linalg.triangular_solve(L_dV, JT_nm1, lower=False)
            # K,K
            post_Gamma_nm1 = post_Gamma__nm1 + tf.linalg.matmul(JT_nm1, tf.linalg.matmul(post_Gamma_n - prior_Gamma__n, JT_nm1), transpose_a=True)
            #tf.linalg.matmul(LTJT, LTJT, transpose_a=True)
            # K,K
            post_Gamma_inter_nm1 = tf.linalg.matmul(
                post_Gamma__nm1 + tf.linalg.matmul(JT_nm1, post_Gamma_inter_n - post_Gamma__nm1, transpose_a=True),
                JT_nm2)
            return [nm1 - 1, post_mu_nm1, post_Gamma_nm1, post_Gamma_inter_nm1, post_mu_ta.write(place_nm1, post_mu_nm1),
                    post_Gamma_ta.write(place_nm1, post_Gamma_nm1),
                    post_Gamma_inter_ta.write(place_nm1, post_Gamma_inter_nm1)]

        post_mu_ta = tf.TensorArray(float_type, size=B)
        post_Gamma_ta = tf.TensorArray(float_type, size=B)
        post_Gamma_inter_ta = tf.TensorArray(float_type, size=B)

        post_mu_0 = post_mu_[-1, :]
        post_Gamma_0 = post_Gamma_[-1, :, :]
        # K, K
        post_Gamma_inter_0 = tf.linalg.matmul(tf.eye(K,dtype=float_type) - last_KX, post_Gamma_0)

        post_mu_ta = post_mu_ta.write(0, post_mu_0)
        post_Gamma_ta = post_Gamma_ta.write(0, post_Gamma_0)
        post_Gamma_inter_ta = post_Gamma_inter_ta.write(0, post_Gamma_inter_0)

        [_, _, _, _,
         post_mu_ta, post_Gamma_ta, post_Gamma_inter_ta] = \
            tf.while_loop(cond,
                          body,
                          [B - 2,
                           post_mu_0,
                           post_Gamma_0,
                           post_Gamma_inter_0,
                           post_mu_ta,
                           post_Gamma_ta,
                           post_Gamma_inter_ta])

        post_mu = post_mu_ta.stack()[::-1, ...]
        post_Gamma = post_Gamma_ta.stack()[::-1, ...]
        post_Gamma_inter = post_Gamma_inter_ta.stack()[::-1, ...]

        return post_mu, post_Gamma, post_Gamma_inter

    def forward_filter(self, y, X, mu_0, Gamma_0, Sigma, Omega):
        """
        Perform a forward filter pass
        :param y: tf.Tensor
            [B, N]
        :param X: tf.Tensor
            [N, K]
        :param mu_0: tf.Tensor
            [K]
        :param Gamma_0: tf.Tensor
            [K,K]
        :param Sigma: tf.Tensor
            [N, N]
        :param Omega: tf.Tensor
            [K,K]
        :return: tuple
            prior_Gamma [B, K, K]
            post_mu [B,K]
            post_Gamma [B, K, K]
            last_KX [K, K]
        """
        B = tf.shape(y)[0]
        N = tf.shape(y)[1]
        K = tf.shape(X)[1]


        def cond(n, *args):
            return n < B

        def body(n, post_mu_n1, post_Gamma_n1, KX_n1, prior_Gamma_ta, post_mu_ta, post_Gamma_ta):
            ###
            # get prior
            # K
            prior_mu_n = post_mu_n1
            # K,K
            prior_Gamma_n = post_Gamma_n1 + Omega
            ###
            # get posterior
            # N, K
            XGamma = tf.matmul(X, prior_Gamma_n)
            # N, N
            XGammaXT = tf.matmul(XGamma, X, transpose_b=True)
            # N, K
            KT = tf.linalg.lstsq(Sigma + XGammaXT + diagonal_jitter(N), XGamma, fast=False)
            # [N,K].[K]->[N]
            Xmu = tf.linalg.matmul(X, prior_mu_n[:, None])[:, 0]
            # [N,K].[N,1]->[K]
            post_mu_n = prior_mu_n + tf.linalg.matmul(KT, (y[n, :] - Xmu)[:, None], transpose_a=True)[:, 0]
            # [N,K].[ N,K]->[ K, K]
            KX = tf.linalg.matmul(KT, X, transpose_a=True)
            # K,K
            post_Gamma_n = tf.linalg.matmul(tf.eye(K, dtype=float_type) - KX, prior_Gamma_n)

            return [n + 1, post_mu_n, post_Gamma_n, KX, prior_Gamma_ta.write(n, prior_Gamma_n),
                    post_mu_ta.write(n, post_mu_n), post_Gamma_ta.write(n, post_Gamma_n)]

        prior_Gamma_ta = tf.TensorArray(float_type, size=B)
        post_mu_ta = tf.TensorArray(float_type, size=B)
        post_Gamma_ta = tf.TensorArray(float_type, size=B)

        # K, K
        dummy_KX = tf.zeros([K, K], dtype=float_type)

        [_, _, _, last_KX,
         prior_Gamma_ta, post_mu_ta, post_Gamma_ta] = \
            tf.while_loop(cond,
                          body,
                          [tf.constant(0, tf.int32),
                           mu_0,
                           Gamma_0,
                           dummy_KX,
                           prior_Gamma_ta,
                           post_mu_ta,
                           post_Gamma_ta])

        prior_Gamma = prior_Gamma_ta.stack()
        post_mu = post_mu_ta.stack()
        post_Gamma = post_Gamma_ta.stack()

        return prior_Gamma, post_mu, post_Gamma, last_KX

    def run(self, y):
        return self.sess.run(self.result, feed_dict={self.y_pl: y})
