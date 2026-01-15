from tqdm import tqdm, trange
import zarr
import numpy as np
import zarr.codecs
from torch.utils.data import Dataset
import torch

class ZarrRecorder:
    def __init__(self, path):
        self.z_handle = zarr.open(path, mode='w')
        self.arrays = {}
        self.frame_count = 0

    def record_frame(self, obs: dict, action: torch.Tensor):
        merged_obs = {**obs, "action": action}

        def torch_type_to_zarr_type_and_default_value(tensor: torch.Tensor):
            # float32, uint8, int32, int64, bool
            if tensor.dtype == torch.float32:
                return np.float32, 0.0
            elif tensor.dtype == torch.uint8:
                return np.uint8, 0
            elif tensor.dtype == torch.int32:
                return np.int32, 0
            elif tensor.dtype == torch.int64:
                return np.int64, 0
            elif tensor.dtype == torch.bool:
                return np.bool_, False
            else:
                raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")

        for key, observation in merged_obs.items():
            if key not in self.arrays:
                shape = (observation.shape[0], 1048576) + observation.shape[1:]
                chunks = (observation.shape[0], 8) + observation.shape[1:]
                if key == "rgb":
                    chunks = (1, 1) + observation.shape[1:]
                if key == "done":
                    chunks = (observation.shape[0], 128) + observation.shape[1:]
                dtype, fill_value = torch_type_to_zarr_type_and_default_value(observation)
                self.arrays[key] = self.z_handle.create_array(
                    key,
                    shape=shape,
                    chunks=chunks,
                    dtype=dtype,
                    fill_value=fill_value,
                    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle)
                )
            self.arrays[key][:, self.frame_count, ...] = observation.cpu().numpy()
        self.frame_count += 1

            