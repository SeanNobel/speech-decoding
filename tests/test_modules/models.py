import torch
import torch.nn as nn

from speech_decoding.models import SpatialAttention


class TestSubjectBlock(nn.Module):
    def __init__(self, args):
        super(TestSubjectBlock, self).__init__()

        self.num_subjects = args.num_subjects
        self.D1 = args.D1
        self.K = args.K
        self.spatial_attention = SpatialAttention(args)
        self.conv = nn.Conv1d(in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1)

        # NOTE: The below implementations are equivalent to learning a matrix:
        self.subject_matrix = nn.Parameter(torch.rand(self.num_subjects, self.D1, self.D1))
        # self.subject_layer = [
        #     nn.Conv1d(in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1, device=device)
        #     for _ in range(self.num_subjects)
        # ]

    def forward(self, X, subject_idxs):
        X = self.spatial_attention(X)  # ( B, 270, 256 )
        X = self.conv(X)  # ( B, 270, 256 )

        # NOTE to Sensho: this has caused problems. I slighly changed it here. Hope it doesn't break anything for you
        _subject_idxs = subject_idxs.tolist()
        X = (
            self.subject_matrix[_subject_idxs] @ X
        )  # ( 270, 270 ) @ ( B , 270, 256 ) -> ( B, 270, 256 )
        # _X = []
        # for i, x in enumerate(X):  # x: ( 270, 256 )
        #     x = self.subject_layer[subject_idxs[i]](x.unsqueeze(0))  # ( 1, 270, 256 )
        #     _X.append(x.squeeze())
        # X = torch.stack(_X)

        return X  # ( B, 270, 256 )
