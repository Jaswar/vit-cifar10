import torch as th
import torch.nn.functional as F


class SpatialTransformer(th.nn.Module):

    def __init__(self, num_embeddings, embedding_size, hidden_dim=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_embeddings = num_embeddings - 1  # one less due to class token
        self.embedding_size = embedding_size
        self.embedding_width = int(round(embedding_size ** 0.5))
        self.hidden_dim = hidden_dim
        assert self.embedding_width * self.embedding_width == self.embedding_size

        self.localization = th.nn.Sequential(th.nn.Conv2d(in_channels=self.num_embeddings, out_channels=64, kernel_size=3),
                                             th.nn.MaxPool2d(2, 2),
                                             th.nn.ReLU(),
                                             th.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                                             th.nn.MaxPool2d(2, 2),
                                             th.nn.ReLU(),
                                             th.nn.Flatten(),
                                             th.nn.Linear(64 * (((self.embedding_width - 2) // 2 - 2) // 2) *
                                                          (((self.embedding_width - 2) // 2 - 2) // 2), hidden_dim),
                                             th.nn.ReLU(),
                                             th.nn.Linear(hidden_dim, 2 * 3))

    def forward(self, x):
        batch_size = x.shape[0]

        out = x[:, 1:, :]  # remove the class token
        out = out.view(batch_size, self.num_embeddings, self.embedding_width, self.embedding_width)  # view as images

        theta = self.localization(out)
        theta = theta.view(batch_size, 2, 3)

        grid = F.affine_grid(theta, out.shape, align_corners=True)
        out = F.grid_sample(out, grid, align_corners=True)

        out = out.view(batch_size, self.num_embeddings, self.embedding_size)  # view back as embeddings
        out = th.cat([x[:, :1, :], out], dim=1)  # add the class token back
        return out
