import torch as t
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim, bias=False)
        self.fc2 = nn.Linear(latent_dim, 1, bias=False)

    def forward(self, X):
        assert X.ndim == 3
        X = self.fc1(X).tanh()
        X = self.fc2(X).softmax(dim=1)
        return X


class AttentionMIL(nn.Module):

    def __init__(self, encoder, attention, classifier):
        super().__init__()
        self.encoder    = encoder
        self.attention  = attention
        self.classifier = classifier

    def encode(self, X):
        BatchSize, BagSize = X.shape[:2]
        DataShape = X.shape[2:]

        X = X.reshape((BatchSize * BagSize), *DataShape)
        X = self.encoder(X)
        assert X.ndim == 2, X.shape
        X = X.reshape((BatchSize, BagSize, -1))
        return X

    def assess(self, H):
        assert H.ndim == 3, H.shape
        BatchSize, BagSize, FeatureDim = H.shape
        A = self.attention(H)
        assert A.shape == (BatchSize, BagSize, 1), A.shape
        return A

    def centroids(self, H, A):
        BatchSize, BagSize, FeatureDim = H.shape
        H = t.bmm(t.permute(A, (0, 2, 1)), H)
        assert H.shape == (BatchSize, 1, FeatureDim), H.shape
        return H

    def classify(self, H):
        BatchSize, BagSize, FeatureDim = H.shape
        H = H.reshape((BatchSize * BagSize, FeatureDim))
        Y = self.classifier(H)
        return Y.reshape((BatchSize, BagSize, -1))

