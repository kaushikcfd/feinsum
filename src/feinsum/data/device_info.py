from collections.abc import Mapping

# Mapping from device name to peak flop rate in GFlOps/s
DEV_TO_PEAK_GFLOPS: Mapping[str, Mapping[str, float]]
DEV_TO_PEAK_GFLOPS = {
    "NVIDIA TITAN V": {
        "float32": 12288,
        "float64": 6144,
    },
    "NVIDIA GeForce GTX 1650": {
        "float32": 3916.0,
        "float64": 122.4,
    },
}

# Mapping from device name to peak bandwidth in GB/s
DEV_TO_PEAK_BW = {
    "NVIDIA TITAN V": 652.8,
    "NVIDIA GeForce GTX 1650": 192.0,
}
