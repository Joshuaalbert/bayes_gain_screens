import tensorflow.compat.v1 as tf
float_type = tf.float64


class NLDSSmoother(object):
    def __init__(self, num_latent, num_observables, update, momentum=0.5, serve_shapes=None, session:tf.Session=None):
        """
        Perform non-linear dynamics smoothing.
        :param serve_shapes: dict of shapes of kwargs that will be sliced and served
            Each value is assumed to be [...] and will form a [B, ...] placeholder
            and will be sliced on first dimension and passed to _update function.
        """
        N = num_observables
        K = num_latent
        self._update = update

        if serve_shapes is None:
            serve_shapes = []
        if session is None:
            graph = tf.Graph()
            self.sess = tf.Session(graph=graph)
        else:
            graph = session.graph
            self.sess = session

        with graph.as_default():
            with tf.name_scope('nlds_smoother'):
                Nmax_pl = tf.placeholder(tf.int32, shape=(), name='Nmax')
                # B, N
                y_pl = tf.placeholder(float_type, shape=[None, N], name='y')
                B = tf.shape(y_pl)[0]
                # K,K
                # Omega_0_pl = tf.placeholder(float_type, shape=[K, K], name='Omega_0')
                Omega_0_pl = tf.placeholder(float_type, shape=None, name='Omega_0')
                Omega_0 = tf.broadcast_to(Omega_0_pl, [B, K, K])

                mu_0_pl = tf.placeholder(float_type, shape=[K], name='mu_0')
                Gamma_0_pl = tf.placeholder(float_type, shape=[K, K], name='Gamma_0')

                serve_pl = [tf.placeholder(float_type, shape=[None] + list(shape)) for shape in serve_shapes]

                # N,N
                # Sigma_0_pl = tf.placeholder(float_type, shape=[N, N], name='Sigma_0')
                Sigma_0_pl = tf.placeholder(float_type, shape=None, name='Sigma_0')
                Sigma_0 = tf.broadcast_to(Sigma_0_pl, [B, N, N])

                ###
                # Bayesian evidence

                def cond(n, *args):
                    return n < Nmax_pl

                def body(n, mu_0_n1, Gamma_0_n1, Sigma_n1, Omega_n1, post_mu_n1, post_Gamma_n1):

                    prior_Gamma, post_mu_f, post_Gamma_f = self.forward_filter(y_pl, mu_0_n1, Gamma_0_n1,
                                                                               Sigma_n1, Omega_n1, serve_pl)

                    post_mu_b, post_Gamma_b, post_Gamma_inter = self.backward_filter(prior_Gamma, post_mu_f,
                                                                                     post_Gamma_f, Omega_n1)

                    res = self.parameter_estimation(y_pl, post_mu_b, post_Gamma_b, post_Gamma_inter, serve_pl)

                    Sigma = momentum * Sigma_n1 + (1. - momentum) * res['R_new']
                    Omega = momentum * Omega_n1 + (1. - momentum) * res['Q_new']

                    mu_0 = momentum*mu_0_n1 + (1. - momentum)*res['mu_0']#mu_0_n1#
                    Gamma_0 = momentum * Gamma_0_n1 + (1. - momentum) * res['Gamma_0']#Gamma_0_n1#

                    post_Gamma_b.set_shape(post_Gamma_n1.shape)
                    post_mu_b.set_shape(post_mu_n1.shape)

                    Omega.set_shape(Omega_n1.shape)
                    Sigma.set_shape(Sigma_n1.shape)

                    return [n + 1, mu_0, Gamma_0, Sigma, Omega, post_mu_b, post_Gamma_b]

                [_, mu_0, Gamma_0, Sigma, Omega, post_mu, post_Gamma] = \
                    tf.while_loop(cond,
                                  body,
                                  [tf.constant(0, tf.int32),
                                   mu_0_pl,
                                   Gamma_0_pl,
                                   Sigma_0,
                                   Omega_0,
                                   tf.zeros([B, K], float_type),
                                   tf.zeros([B, K, K], float_type)
                                   ],
                                  name='iteration_while_loop')

                self.post_y_mean, self.post_y_cov = self._update.predictive_distribution(post_mu, post_Gamma)
                self.y_pl = y_pl
                self.Sigma_0_pl = Sigma_0_pl
                self.Omega_0_pl = Omega_0_pl
                self.mu_0_pl = mu_0_pl
                self.Gamma_0_pl = Gamma_0_pl
                self.Nmax_pl = Nmax_pl
                self.serve_pl = serve_pl
                self.result = dict(
                    post_mu=post_mu,
                    post_Gamma=post_Gamma,
                    Omega=Omega,
                    Sigma=Sigma,
                    # mu_0=mu_0,
                    # Gamma_0=Gamma_0,
                    # post_y_mean=self.post_y_mean,
                    # post_y_cov=self.post_y_cov
                )

                tf.summary.image('Sigma',tf.linalg.diag_part(Sigma)[None,:,:,None])
                tf.summary.image('Omega',tf.linalg.diag_part(Omega)[None,:,:,None])
                self.summary = tf.summary.merge_all()

    def parameter_estimation(self, y, post_mu_b, post_Gamma_b, post_Gamma_inter, serve_values):
        """
        M-step
        :param y: [B, N]
        :param post_mu: [B, K]
        :param Pt: [B, K, K]
        :param Ptt1: [B, K, K]
        :return:
        """
        with tf.name_scope('parameter_estimation'):
            B = tf.shape(y)[0]
            N = tf.shape(y)[1]
            K = tf.shape(post_Gamma_b)[2]

            ###
            # observation covariance estimate
            # S, B, N
            R_new, Q_new = self._update.get_params(y, post_mu_b, post_Gamma_b, *serve_values)

            mu_0 = post_mu_b[0, :]
            Gamma_0 = post_Gamma_b[0, :, :]

            return dict(R_new=R_new,
                        Q_new=Q_new,
                        mu_0=mu_0,
                        Gamma_0=Gamma_0)

    def backward_filter(self, prior_Gamma, post_mu_f, post_Gamma_f, Omega):
        """

        http://mlg.eng.cam.ac.uk/zoubin/course04/tr-96-2.pdf

        Perform a forward filter pass
        :param A: tf.Tensor
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
        with tf.name_scope('backward_filter'):
            B = tf.shape(post_Gamma_f)[0]
            K = tf.shape(post_mu_f)[1]

            # JT_aug shifted so that n accesses n-1
            # B,K,K
            # J'_t = (V_t+1^t)^-1 . V_t^t
            JT_aug = tf.concat([tf.zeros([1, K, K], dtype=float_type),
                                tf.linalg.lstsq(prior_Gamma[1:, :, :], post_Gamma_f[:-1, :, :], fast=False)],
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

                prior_Gamma_n = prior_Gamma[n, :, :]
                post_Gamma_f_nm1 = post_Gamma_f[nm1, :, :]
                post_mu_f_nm1 = post_mu_f[nm1, :]

                # K,K
                JT_nm1 = JT_aug[nm1 + 1, :, :]
                JT_nm2 = JT_aug[nm2 + 1, :, :]

                # K
                post_mu_nm1 = post_mu_f_nm1 + \
                              tf.linalg.matmul(JT_nm1, (post_mu_n - post_mu_f_nm1)[:, None],
                                               transpose_a=True)[:, 0]

                # K,K
                post_Gamma_nm1 = post_Gamma_f_nm1 + tf.linalg.matmul(JT_nm1,
                                                                     tf.linalg.matmul(post_Gamma_n - prior_Gamma_n, JT_nm1),
                                                                     transpose_a=True)

                # K,K
                post_Gamma_inter_nm1 = tf.linalg.matmul(
                    post_Gamma_f_nm1 + tf.linalg.matmul(JT_nm1, post_Gamma_inter_n - post_Gamma_f_nm1, transpose_a=True),
                    JT_nm2)

                return [nm1 - 1, post_mu_nm1, post_Gamma_nm1, post_Gamma_inter_nm1,
                        post_mu_ta.write(place_nm1, post_mu_nm1),
                        post_Gamma_ta.write(place_nm1, post_Gamma_nm1),
                        post_Gamma_inter_ta.write(place_nm1, post_Gamma_inter_nm1)]

            post_mu_ta = tf.TensorArray(float_type, size=B)
            post_Gamma_ta = tf.TensorArray(float_type, size=B)
            post_Gamma_inter_ta = tf.TensorArray(float_type, size=B)

            post_mu_B = post_mu_f[-1, :]
            post_Gamma_B = post_Gamma_f[-1, :, :]
            # K, K
            post_Gamma_inter_B = tf.linalg.lstsq(
                tf.eye(K, dtype=float_type) + tf.linalg.lstsq(post_Gamma_f[-1, :, :], Omega[-1,:,:], fast=False) - tf.linalg.lstsq(
                    prior_Gamma[-1, :, :], Omega[-1,:,:], fast=False),
                post_Gamma_f[-2, :, :], fast=False)

            post_mu_ta = post_mu_ta.write(0, post_mu_B)
            post_Gamma_ta = post_Gamma_ta.write(0, post_Gamma_B)
            post_Gamma_inter_ta = post_Gamma_inter_ta.write(0, post_Gamma_inter_B)

            [_, _, _, _,
             post_mu_ta, post_Gamma_ta, post_Gamma_inter_ta] = \
                tf.while_loop(cond,
                              body,
                              [B - 2,
                               post_mu_B,
                               post_Gamma_B,
                               post_Gamma_inter_B,
                               post_mu_ta,
                               post_Gamma_ta,
                               post_Gamma_inter_ta],
                              name='backwards_while_loop')

            post_mu = post_mu_ta.stack()[::-1, ...]
            post_Gamma = post_Gamma_ta.stack()[::-1, ...]
            post_Gamma_inter = post_Gamma_inter_ta.stack()[::-1, ...]

            return post_mu, post_Gamma, post_Gamma_inter

    def forward_filter(self, y, mu_0, Gamma_0, Sigma, Omega, serve_pl):
        with tf.name_scope('forward_filter'):
            B = tf.shape(y)[0]

            def cond(n, *args):
                return n < B

            def body(n, post_mu_n1, post_Gamma_n1, prior_Gamma_ta, post_mu_ta, post_Gamma_ta):
                ###
                # get prior
                prior_mu_n = post_mu_n1
                # K,K
                prior_Gamma_n = post_Gamma_n1 + Omega[n, :, :]

                sliced_serve_pl = [v[n, ...] for v in serve_pl]

                post_mu_n, post_Gamma_n = self._update(n, prior_mu_n, prior_Gamma_n, y[n, :], Sigma[n, :, :], *sliced_serve_pl)
                post_mu_n.set_shape(post_mu_n1.shape)
                post_Gamma_n.set_shape(post_Gamma_n1.shape)
                return [n + 1, post_mu_n, post_Gamma_n, prior_Gamma_ta.write(n, prior_Gamma_n),
                        post_mu_ta.write(n, post_mu_n), post_Gamma_ta.write(n, post_Gamma_n)]

            prior_Gamma_ta = tf.TensorArray(float_type, size=B)
            post_mu_ta = tf.TensorArray(float_type, size=B)
            post_Gamma_ta = tf.TensorArray(float_type, size=B)

            [_, _, _,
             prior_Gamma_ta, post_mu_ta, post_Gamma_ta] = \
                tf.while_loop(cond,
                              body,
                              [tf.constant(0, tf.int32),
                               mu_0,
                               Gamma_0,
                               prior_Gamma_ta,
                               post_mu_ta,
                               post_Gamma_ta],
                              name='forwards_while_loop')

            prior_Gamma = prior_Gamma_ta.stack()
            post_mu = post_mu_ta.stack()
            post_Gamma = post_Gamma_ta.stack()

            return prior_Gamma, post_mu, post_Gamma

    def run(self, y, Sigma_0, Omega_0, mu_0, Gamma_0, Nmax=2, serve_values=None, logdir=None, step=0):
        if serve_values is not None:
            feed_dict = {v_pl: v for (v, v_pl) in zip(serve_values, self.serve_pl)}
        else:
            feed_dict = {}

        feed_dict.update({self.y_pl: y,
                          self.mu_0_pl: mu_0,
                          self.Gamma_0_pl: Gamma_0,
                          self.Sigma_0_pl: Sigma_0,
                          self.Omega_0_pl: Omega_0,
                          self.Nmax_pl: Nmax})

        if logdir is not None:
            writer = tf.summary.FileWriter(logdir, self.sess.graph, session=self.sess)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            res, summary_eval = self.sess.run([self.result, self.summary], feed_dict=feed_dict,
                                            options=run_options,
                                            run_metadata=run_metadata
                                            )
            writer.add_run_metadata(run_metadata, 'step{}'.format(step))
            writer.add_summary(summary_eval, step)
            return res
        else:
            return self.sess.run(self.result, feed_dict=feed_dict)
