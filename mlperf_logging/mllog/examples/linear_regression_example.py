# Copyright 2019 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import time
import numpy as np
from jax import numpy as jnp
import jax
from mlperf_logging import mllog
import mlperf_logging.mllog.constants as mllog_constants

MLLOGGER = mllog.get_mllogger()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-log", type=str, default="result_0.txt")
    parser.add_argument("--output-folder", type=str, default="output")
    parser.add_argument("--n-processes", type=int, default=8)
    parser.add_argument("--measurement-type", choices=["DC", "AC"], default="DC")
    parser.add_argument("--convertion-eff", type=float, default=1.0)

    # Dummy dataset arguments
    parser.add_argument("--size", type=int, default=1048576)
    parser.add_argument("--in-features", type=int, default=100)
    parser.add_argument("--out-features", type=int, default=1)
    parser.add_argument("--eval-size", type=float, default=0.05)

    # Dataloader arguments
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shuffle", type=bool, default=True)

    # Train arguments
    parser.add_argument("--epochs", type=int, default=10)

    # Other arguments
    parser.add_argument("--seed", type=int, required=False, default=None)
    parser.add_argument("--min-time", type=int, required=False, default=None)
    args = parser.parse_args()
    return args


class Dataset:
    def __init__(
        self,
        in_features,
        out_features,
        size,
        batch_size,
        shuffle,
        noise_scale=0.1,
        name="train_dataset",
        hidden_weigths=None,
        hidden_bias=None,
    ) -> None:
        self.shuffle = shuffle
        self.size = size

        if self.shuffle:
            self.p = np.random.permutation(np.arange(self.size))
        else:
            self.p = jnp.arange(size)

        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features

        if hidden_weigths is not None:
            self.hidden_weigths = hidden_weigths
        else:
            hidden_weigths = np.random.randn(in_features, out_features)
        if hidden_bias is not None:
            self.hidden_bias = hidden_bias
        else:
            hidden_bias = np.random.randn(1, out_features)

        self.x, self.y = self.generate_dummy_dataset(
            in_features, out_features, size, noise_scale
        )

        self.name = name
        self.log_initialization()
        self.next = 0

    def generate_dummy_dataset(self, in_features, out_features, size, noise_scale=0.1):
        noise = np.random.randn(size, out_features) * noise_scale
        x = np.random.randn(size, in_features)
        y = (np.dot(x, self.hidden_weigths) + self.hidden_bias) + noise
        return x, y

    def get_batch(self):
        inf = self.next
        sup = min(self.next + self.batch_size, len(self.p))
        batch_x, batch_y = self.x[self.p[inf:sup]], self.y[self.p[inf:sup]]
        self.next += self.batch_size
        return batch_x, batch_y

    def has_next(self):
        return self.next < len(self.p)

    def reset(self):
        self.next = 0
        if self.shuffle:
            self.p = np.random.permutation(np.arange(self.size))

    def log_initialization(self):
        MLLOGGER.event(key="dataset_name", value=self.name)
        MLLOGGER.event(key="dataset_size", value=self.size)
        MLLOGGER.event(key="dataset_input_features", value=self.in_features)
        MLLOGGER.event(key="dataset_output_features", value=self.out_features)
        MLLOGGER.event(key="dataset_batch_size", value=self.batch_size)
        MLLOGGER.event(key="dataset_shuffle", value=self.shuffle)


class Model:
    def __init__(self, in_features, out_features, **kwargs) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.w = np.random.randn(in_features, out_features)
        self.b = np.random.randn(1, out_features)
        self.log_initialization()
        self.optimizer = SGDOptimizer(self._loss_fn, [0, 1], **kwargs)

    def _forward(self, w, b, x):
        return jnp.dot(x, w) + b

    def forward(self, x):
        assert x.shape[1] == self.in_features
        return self._forward(self.w, self.b, x)

    def _loss_fn(self, w, b, x, y):
        n = x.shape[0]
        y_pred = jnp.dot(x, w) + b
        mse = jnp.sum(jnp.square(y - y_pred)) / (2 * n)
        return mse

    def backward(self, x, y):
        self.optimizer.compute_gradient(self.w, self.b, x, y)
        self.w, self.b = self.optimizer.backward_step(self.w, self.b)
        return self._loss_fn(self.w, self.b, x, y)

    def log_initialization(self):
        MLLOGGER.event(
            key=mllog_constants.WEIGHTS_INITIALIZATION, metadata=dict(tensor="w")
        )
        MLLOGGER.event(
            key=mllog_constants.WEIGHTS_INITIALIZATION, metadata=dict(tensor="b")
        )


class SGDOptimizer:
    def __init__(self, loss_fn, argnums, **kwargs) -> None:
        self.grad_f = jax.grad(loss_fn, argnums)
        self.gradients = 0
        self.lr = kwargs.get("lr", 0.01)
        self.l2 = kwargs.get("l2", 0.0)
        self.log_initialization()

    def compute_gradient(self, w, b, x, y):
        self.gradients = self.grad_f(w, b, x, y)
        return self.gradients

    def backward_step(self, w, b):
        w *= 1 - self.l2
        b *= 1 - self.l2
        w -= self.lr * self.gradients[0]
        b -= self.lr * self.gradients[1]
        return w, b

    def log_initialization(self):
        MLLOGGER.event(key="opt_lr", value=self.lr)
        MLLOGGER.event(key="opt_l2", value=self.l2)


class MSEMetric:
    def __init__(self) -> None:
        # Compute the MSEMetric in batches
        self.count = 0
        self.mse = 0
        pass

    def update(self, y_true, y_pred):
        self.count += y_true.shape[0]
        self.mse += jnp.sum(jnp.square(y_true - y_pred))

    def reset(self):
        self.count = 0
        self.mse = 0
        pass

    def result(self):
        if self.count == 0:
            return 0
        else:
            return float(self.mse / self.count)


def load_datasets(args):
    in_features, out_features, size, eval_size = (
        args.in_features,
        args.out_features,
        args.size,
        args.eval_size,
    )
    batch_size, shuffle = args.batch_size, args.shuffle
    train_size = int(size * (1 - eval_size))
    hidden_weights = np.random.randn(in_features, out_features)
    hidden_bias = np.random.randn(1, out_features)
    train_dataset = Dataset(
        in_features,
        out_features,
        train_size,
        batch_size,
        shuffle,
        hidden_weigths=hidden_weights,
        hidden_bias=hidden_bias,
    )
    eval_dataset = Dataset(
        in_features,
        out_features,
        size - train_size,
        batch_size,
        False,
        name="eval_dataset",
        hidden_weigths=hidden_weights,
        hidden_bias=hidden_bias,
    )

    return train_dataset, eval_dataset


def train(
    model: Model, train_dataset: Dataset, eval_dataset: Dataset, metric: MSEMetric, args
):
    epochs = args.epochs
    # Main loop
    MLLOGGER.start(key=mllog_constants.RUN_START)
    for epoch in range(epochs):
        # Train loop
        MLLOGGER.start(
            key=mllog_constants.EPOCH_START, value=epoch, metadata=dict(epoch_num=epoch)
        )
        while train_dataset.has_next():
            x_batch, y_batch = train_dataset.get_batch()
            loss = model.backward(x_batch, y_batch)
        train_dataset.reset()
        MLLOGGER.event(
            key=mllog_constants.EPOCH_STOP, value=epoch, metadata=dict(epoch_num=epoch)
        )

        # Evaluation loop
        MLLOGGER.start(
            key=mllog_constants.EVAL_START, value=epoch, metadata=dict(epoch_num=epoch)
        )
        while eval_dataset.has_next():
            x_batch, y_batch = eval_dataset.get_batch()
            y_pred = model.forward(x_batch)
            metric.update(y_batch, y_pred)
        MLLOGGER.event(
            key=mllog_constants.EVAL_ACCURACY,
            value=metric.result(),
            metadata=dict(epoch_num=epoch),
        )
        metric.reset()
        MLLOGGER.event(
            key=mllog_constants.EVAL_STOP, value=epoch, metadata=dict(epoch_num=epoch)
        )

        eval_dataset.reset()

    MLLOGGER.event(key=mllog_constants.RUN_STOP, metadata=dict(status="success"))


def run():
    args = get_args()
    mllog.config(filename=f"{args.output_folder}/{args.perf_log}")
    start_time = time.time()

    # Initialization
    MLLOGGER.start(key=mllog_constants.INIT_START)
    np.random.seed(args.seed)
    MLLOGGER.event(key=mllog_constants.SEED, value=args.seed)
    train_dataset, eval_dataset = load_datasets(args)
    model = Model(args.in_features, args.out_features)
    metric = MSEMetric()
    MLLOGGER.end(key=mllog_constants.INIT_STOP)
    train(model, train_dataset, eval_dataset, metric, args)

    if args.min_time is not None:
        # Remove handler to avoid logging the result in file
        mllog.mllogger.logger.removeHandler(mllog.mllogger.logger.handlers[1])
        while (time.time() - start_time) < args.min_time:
            MLLOGGER.event(key="Time passed", value=(time.time() - start_time))
            # Uncomment line to reset model and start over training
            # model = Model(args.in_features, args.out_features)
            train(model, train_dataset, eval_dataset, metric, args)


if __name__ == "__main__":
    run()
