from hydra import initialize, compose
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict

from speech_decoding.models import *
from tests.test_modules.models import *


def test_spatial_attention() -> None:
    with initialize(version_base=None, config_path="../configs"):
        args = compose(config_name="config")
        with open_dict(args):
            # FIXME: get_original_cwd() can't be called
            args.root_dir = "/home/sensho/speech_decoding"  # get_original_cwd()

        input = torch.rand(8, 208, 256).to(device)
        output = SpatialAttention(args)(input)
        output_test = SpatialAttentionTest2(args)(input)

        assert output == output_test
