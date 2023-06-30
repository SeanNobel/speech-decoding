import torch
import torchaudio.functional as F

from termcolor import cprint


class SpeechDecodingDatasetBase(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

    def _resample_audio(
        self, waveform: torch.Tensor, sample_rate: int, resample_rate: int = 16000
    ) -> torch.Tensor:
        """Resamples audio to 16kHz. (16kHz is required by wav2vec2.0)"""

        waveform = F.resample(
            waveform,
            sample_rate,
            resample_rate,
            lowpass_filter_width=self.lowpass_filter_width,
        )

        len_audio_s = waveform.shape[1] / resample_rate
        cprint(f">>> Audio length: {len_audio_s} s.", color="cyan")

        return waveform
