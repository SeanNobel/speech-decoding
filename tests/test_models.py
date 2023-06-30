from hydra import initialize, compose
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict

from speech_decoding.models import *
from tests.modules_for_test.models import *

torch.manual_seed(0)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

with initialize(version_base=None, config_path="../configs/"):
    args = compose(config_name="config")

    with open_dict(args):
        args.root_dir = "/home/sensho/speech_decoding"


def test_spatial_attention() -> None:
    # NOTE: 60 channels is for Brennan2018
    input = torch.rand(8, 60, 360).to(device)

    sa = SpatialAttention(args).eval().to(device)

    z_re = sa.z.real.reshape(args.D1, args.K, args.K)
    z_im = sa.z.imag.reshape(args.D1, args.K, args.K)

    sa_test1 = SpatialAttentionTest1(args, z_re, z_im).eval().to(device)
    sa_test2 = SpatialAttentionTest2(args, z_re, z_im).eval().to(device)

    output = sa(input)
    output_test1 = sa_test1(input)
    output_test2 = sa_test2(input)

    assert torch.allclose(output, output_test1, rtol=1e-4, atol=1e-5)
    assert torch.allclose(output, output_test2)


def test_classifier():
    classifier = Classifier()

    Y = torch.rand(64, 512, 90)
    Z = torch.rand(64, 512, 90)

    _, _, similarity_train = classifier(Z, Y)
    _, _, similarity_test = classifier(Z, Y, sequential=True)

    assert torch.allclose(similarity_train, similarity_test)


# def test_standard_normalization():
#     input = torch.rand(64, 1024, 360)
#     output = BrainEncoder._standard_normalization(input)

#     input0_norm = (input[0] - input[0].mean()) / input[0].std()

#     assert torch.equal(output[0], input0_norm)
