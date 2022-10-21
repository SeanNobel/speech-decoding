import mne, mne_bids
import numpy as np
import torch


def ch_locations_2d(dataset_name):
    if dataset_name == "Brennan2018":
        montage = mne.channels.make_standard_montage("easycap-M10")
        info = mne.create_info(ch_names=montage.ch_names,
                               sfreq=512.,
                               ch_types="eeg")
        info.set_montage(montage)

        layout = mne.channels.find_layout(info, ch_type="eeg")

        loc = layout.pos[:, :2]  # ( 61, 2 )
        # Channel 29 was broken in Brennan 2018
        loc = np.delete(loc, 28, axis=0)  # ( 60, 2 )

    elif dataset_name == "Gwilliams2022":
        bids_path = mne_bids.BIDSPath(subject='01',
                                      session='0',
                                      task='0',
                                      datatype="meg",
                                      root='data/Gwilliams2022/')
        raw = mne_bids.read_raw_bids(bids_path)

        layout = mne.channels.find_layout(raw.info, ch_type="meg")

        loc = layout.pos[:, :2]

    else:
        raise ValueError()

    # min-max normalization
    loc = (loc - loc.min(axis=0)) / (loc.max(axis=0) - loc.min(axis=0))

    # NOTE: the paper says "In practice, as a_j is periodic, we scale down (x,y) to keep
    # a margin of 0.1 on each side."
    loc = loc * 0.8 + 0.1

    return torch.from_numpy(loc.astype(np.float32))


if __name__ == "__main__":

    layout = ch_locations_2d("Gwilliams2022")

    print(layout.shape)