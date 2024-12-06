import argparse

import pytorch_lightning as pl
import torch as t
import torch.nn as nn
import torchmetrics as tm

from mnist import MNISTBags
from amil import AttentionMIL
from amil import Attention
from lenet import LeNet5


class AttentionMILMNIST(pl.LightningModule):

    def __init__(self):
        super().__init__()

        n_features = 128
        encoder = LeNet5(feature_dim=n_features)
        attention = Attention(input_dim=n_features,
                              latent_dim=n_features // 2)
        classifier = nn.Linear(n_features, 1)

        self.method = AttentionMIL(encoder=encoder,
                                   attention=attention,
                                   classifier=classifier)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        bag_metrics = tm.MetricCollection([tm.Accuracy(task='binary', threshold=0.0),
                                           tm.Precision(task='binary', threshold=0.0),
                                           tm.Recall(task='binary', threshold=0.0),
                                           tm.AveragePrecision(task='binary', threshold=0.0),
                                           tm.AUROC(task='binary', threshold=0.0)])
        self.train_bag_metrics = bag_metrics.clone(prefix='train_bag_')
        self.val_bag_metrics = bag_metrics.clone(prefix='val_bag_')
        self.test_bag_metrics = bag_metrics.clone(prefix='test_bag_')

        inst_metrics = tm.MetricCollection([tm.Accuracy(task='binary', threshold=0.0),
                                            tm.Precision(task='binary', threshold=0.0),
                                            tm.Recall(task='binary', threshold=0.0),
                                            tm.AveragePrecision(task='binary', threshold=0.0),
                                            tm.AUROC(task='binary', threshold=0.0)])
        self.train_inst_metrics = inst_metrics.clone(prefix='train_inst_')
        self.val_inst_metrics = inst_metrics.clone(prefix='val_inst_')
        self.test_inst_metrics = inst_metrics.clone(prefix='test_inst_')

    def training_step(self, batch, batch_idx):
        loss = self._step(batch,
                          self.train_bag_metrics,
                          self.train_inst_metrics)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch,
                          self.val_bag_metrics,
                          self.val_inst_metrics)
        self.log('val_loss', loss, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        loss = self._step(batch,
                          self.test_bag_metrics,
                          self.test_inst_metrics)
        self.log('test_loss', loss, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(),
                                 lr=1e-4,
                                 weight_decay=1e-4)
        scheduler = {
                'scheduler':
                    t.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        patience=16,
                        min_lr=1e-6),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
                }
        return [optimizer], [scheduler]

    def _step(self, batch, bag_metrics, inst_metrics):
        # X ... bag of images (1, N, C, H, W)
        # Y ... bag label (1,)
        X, Y, I = batch
        BatchSize, BagSize = X.shape[:2]
        assert X.ndim == 5 and X.size(0) == 1 and X.size(2) == 1, X.shape
        assert Y.ndim == 1 and Y.shape == (X.size(0),), Y.shape

        H = self.method.encode(X)
        A = self.method.assess(H)
        Z = self.method.centroids(H, A)
        S = self.method.classify(Z).squeeze(2).squeeze(0)
        T = self.method.classify(H)

        loss = self.loss(S, Y.float())

        bag_metrics.update(S.detach(), Y)
        self.log_dict(bag_metrics, on_epoch=True, on_step=False)

        inst_metrics.update(T.detach().squeeze(2).squeeze(0), I.squeeze(0))
        self.log_dict(inst_metrics, on_epoch=True, on_step=False)

        return loss


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--bagsize_mean', type=int, default=10,
                           help='average bag size')
    argparser.add_argument('--bagsize_std', type=int, default=1,
                           help='standard deviation of bag size')
    argparser.add_argument('--train_bags', type=int, default=100,
                           help='number of training bags')
    argparser.add_argument('--seed', type=int, default=42,
                           help='random seed')
    args = argparser.parse_args()

    bagsize_mean   = args.bagsize_mean
    bagsize_std    = args.bagsize_std
    num_train_bags = args.train_bags
    seed           = args.seed

    pl.seed_everything(seed)

    train_data = MNISTBags(root='data',
                           bagsize_mean=bagsize_mean,
                           bagsize_std=bagsize_std,
                           n_bags=num_train_bags,
                           train=True)
    train_loader = t.utils.data.DataLoader(
            train_data,
            batch_size=1,
            batch_sampler=None,
            shuffle=True,
            num_workers=3)

    valid_data = MNISTBags(root='data',
                           bagsize_mean=bagsize_mean,
                           bagsize_std=bagsize_std,
                           n_bags=100,
                           train=True)
    valid_loader = t.utils.data.DataLoader(
            valid_data,
            batch_size=1,
            shuffle=False,
            num_workers=3)

    module = AttentionMILMNIST()
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         max_epochs=1000,
                         min_epochs=50,
                         callbacks=[
                             pl.callbacks.ModelCheckpoint(
                                 save_weights_only=True,
                                 mode="min",
                                 monitor="val_loss"),
                             pl.callbacks.EarlyStopping(
                                 monitor="val_loss",
                                 mode="min",
                                 patience=24
                                 ),
                             ],
                         )
    trainer.fit(module, train_loader, valid_loader)

    test_data = MNISTBags(root='data',
                          bagsize_mean=bagsize_mean,
                          bagsize_std=bagsize_std,
                          n_bags=100,
                          train=False)
    test_loader = t.utils.data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=3)

    trainer.test(ckpt_path='best', dataloaders=test_loader)

