from hydra import initialize, compose
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict

from speech_decoding.models import *
from tests.modules_for_test.models import *

with initialize(version_base=None, config_path="../configs/"):
    args = compose(config_name="config.yaml")


# def test_spatial_attention() -> None:
#     with initialize(version_base=None, config_path="../configs"):
#         args = compose(config_name="config")
#         with open_dict(args):
#             # FIXME: get_original_cwd() can't be called
#             args.root_dir = "/home/sensho/speech_decoding"  # get_original_cwd()

#         input = torch.rand(8, 208, 256).to(device)
#         output = SpatialAttention(args)(input)
#         output_test = SpatialAttentionTest2(args)(input)

#         assert output == output_test


def test_classifier():
    classifier = Classifier()

    Y = torch.rand(64, 512, 90)
    Z = torch.rand(64, 512, 90)

    _, _, similarity_train = classifier(Z, Y)
    _, _, similarity_test = classifier(Z, Y, sequential=True)

    assert torch.allclose(similarity_train, similarity_test)


def test_standard_normalization():
    input = torch.rand(64, 1024, 360)
    output = BrainEncoder._standard_normalization(input)

    input0_norm = (input[0] - input[0].mean()) / input[0].std()

    assert torch.equal(output[0], input0_norm)
