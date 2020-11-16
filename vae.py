from mxnet import gluon
from mxnet.gluon import nn
import mxnet as mx

data_ctx = mx.gpu()
mx.random.seed(1, data_ctx)


class VAE(gluon.HybridBlock):
    def __init__(self, hidden=400, latent=2, layers=1, output=784, batch_size=100, act_type='relu', **kwargs):
        self.soft_zero = 1e-10
        self.n_latent = latent
        self.batch_size = batch_size
        self.output = None
        self.mu = None
        super(VAE, self).__init__(**kwargs)

        with self.name_scope():
            self.encoder = nn.HybridSequential(prefix='encoder')

            for i in range(layers):
                self.encoder.add(nn.Dense(hidden, activation=act_type))
            # noinspection PyTypeChecker
            self.encoder.add(nn.Dense(latent * 2, activation=None))

            self.decoder = nn.HybridSequential(prefix='decoder')
            for i in range(layers):
                self.decoder.add(nn.Dense(hidden, activation=act_type))
            self.decoder.add(nn.Dense(output, activation='sigmoid'))

    def hybrid_forward(self, F, x, **kwargs):
        h = self.encoder(x)
        mu_lv = F.split(h, axis=1, num_outputs=2)
        mu = mu_lv[0]
        lv = mu_lv[1]
        self.mu = mu

        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=data_ctx)
        z = mu + F.exp(0.5 * lv) * eps
        y = self.decoder(z)
        self.output = y

        KL = 0.5 * F.sum(1 + lv - mu * mu - F.exp(lv), axis=1)
        log_loss = F.sum(x * F.log(y + self.soft_zero) + (1 - x) * F.log(1 - y + self.soft_zero), axis=1)
        return -log_loss - KL
