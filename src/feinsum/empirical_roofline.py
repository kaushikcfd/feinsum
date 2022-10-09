import pyopencl as cl
import numpy as np
import loopy as lp
import pyopencl.clrandom as clrandom
import matplotlib.pyplot as plt
from dataclasses import dataclass
import shlex
import subprocess
from subprocess import check_output

# Will get queues for each device with the same name
# since different CL implementations may have different
# performance.


def get_queues_like(queue):
    queues = []
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.name == queue.device.name:
                context = cl.Context(devices=[device])
                queue = cl.CommandQueue(
                    context, properties=cl.command_queue_properties.PROFILING_ENABLE)
                queues.append(queue)
    return queues



@dataclass
class BandwidthTestResult():

    device: str
    tavg: float
    tmin: float
    tmax: float
    bytes_transferred: int
    test_type: str
    test_parameters: tuple = None

    @property
    def avg_bandwidth(self):
        return self.bytes_transferred/self.tavg

    @property
    def max_bandwidth(self):
        return self.bytes_transferred/self.tmin

    @property
    def min_bandwidth(self):
        return self.bytes_transferred/self.tmax

def get_theoretical_maximum_flop_rate(queue, dtype):
    clock_speed = queue.device.max_clock_frequency * 1.0e6
    compute_units = queue.device.max_compute_units
    madd_rate = 2 # Number of multiply add ops per cycle
    try:
        simd_size = queue.device.preferred_work_group_size_multiple
    except cl._cl.LogicError:
        if 'NVIDIA' in queue.device.vendor:
            simd_size = queue.device.warp_size_nv
        elif 'Advanced Micro Devices' in queue.device.vendor:
            simd_size = queue.device.work_group_size_amd

    # For some reason the number of shader units for nvidia
    # is 2*simd_size*compute_units. Probably varies depending
    # on the device.
    simd_multiple = 2 if 'NVIDIA' in queue.device.vendor else 1

    shading_units = compute_units*simd_size*simd_multiple
    
    # May vary depending on the data type
    if dtype == np.float32:
        cycles_per_flop = 1
    elif dtype == np.float64:
        # Can vary depending on the GPU. This only handles
        # the Titan V, V100, A100, MI100, and MI250X
        cycles_per_flop = 1 if "gfx90a" in queue.device.name else 2
    else:
        raise ValueError("Unhandled flop type")

    return (clock_speed*shading_units*madd_rate) // cycles_per_flop 


def get_buffers(queue, dtype_in, n_dtype_in, dtype_out=None,
                n_dtype_out=None, fill_on_device=True):

    if n_dtype_out is None:
        n_dtype_out = n_dtype_in
    if dtype_out is None:
        dtype_out = dtype_in

    n_bytes_in = n_dtype_in*dtype_in().itemsize
    n_bytes_out = n_dtype_out*dtype_out().itemsize
    context = queue.context

    d_out_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY |
                          cl.mem_flags.HOST_NO_ACCESS, size=n_bytes_out)

    if fill_on_device:
        # Requires making a READ_WRITE buffer instead of a READ_ONLY buffer
        if dtype_in in {np.float64, np.float32, np.int32, np.int64}:
            from pyopencl.tools import ImmediateAllocator
            allocator = ImmediateAllocator(queue)
            d_in_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE |
                                 cl.mem_flags.HOST_NO_ACCESS, size=n_bytes_in)
            d_in_buf_arr = cl.array.Array(
                queue, (n_dtype_in,), dtype_in, allocator=allocator, data=d_in_buf)
            clrandom.fill_rand(d_in_buf_arr, queue=queue)
        else:
            raise ValueError(f"Cannot fill array with {dtype_in} on the device")
    else:

        from psutil import virtual_memory

        if np.issubdtype(dtype_in, np.integer):
            if virtual_memory().available < n_bytes_in:
                raise ValueError(
                    "Not enough host memory to fill the buffer from the host")

            max_val = np.iinfo(dtype_in).max
            min_val = np.iinfo(dtype_in).min
            h_in_buf = np.random.randint(
                min_val, high=max_val + 1, size=n_dtype_in, dtype=dtype_in)
        elif np.issubdtype(dtype_in, np.float):
            # The host array is formed as a float64 before being copied and converted
            if virtual_memory().available < \
                        n_dtype_in*(np.float64(0).itemsize + dtype_in().itemsize):
                raise ValueError(
                    "Not enough host memory to fill the buffer from the host")
            h_in_buf = np.random.rand(n_dtype_in).astype(dtype_in)
        else:
            raise ValueError(f"Unsupported dtype: {dtype_in}")

        d_in_buf = cl.Buffer(
            context,
            cl.mem_flags.READ_ONLY |
            cl.mem_flags.HOST_NO_ACCESS |
            cl.mem_flags.COPY_HOST_PTR,
            size=n_bytes_in,
            hostbuf=h_in_buf)

        #TODO: Copy small chunks at a time if the array size is large.
        # Is this actually needed?

        #d_in_buf = cl.Buffer(context,
        #    cl.mem_flags.READ_ONLY |
        #    cl.mem_flags.HOST_WRITE_ONLY |
        #    cl.mem_flags.COPY_HOST_PTR,
        #    size=max_shape_bytes, hostbuf=h_in_buf)
        #for some number of chunks
        #   h_in_buf = ...
        #   evt = cl.enqueue_copy(queue, d_in_buf, h_in_buf) # With offsets

    return d_in_buf, d_out_buf


def get_word_counts(max_shape_dtype, minimum=1):
    word_count_list = []

    word_count = minimum
    # Get some non-multiples of two
    while word_count <= max_shape_dtype:
        word_count_list.append(int(np.floor(word_count)))
        word_count = word_count*1.5
    # Get multiples of two
    for i in range(0, int(np.floor(np.log2(max_shape_dtype)) + 1)):
        if 2**i >= minimum:
            word_count_list.append(2**i)
    word_count_list = sorted(list(set(word_count_list)))
    return word_count_list


def loopy_bandwidth_test_with_queues_like(
        queue, dtype_in=None, fill_on_device=True, fast=True):

    queues = get_queues_like(queue)

    return tuple([loopy_bandwidth_test(q, dtype_in=dtype_in,
                    fill_on_device=fill_on_device, fast=fast) for q in queues])


def loopy_bandwidth_test(queue, n_in_max=None, dtype_in=None, n_out_max=None,
                        dtype_out=None, fill_on_device=True,
                        ntrials=100, fast=True, print_results=False,
                        pollute_caches=True):

    if pollute_caches and n_in_max is not None and n_out_max is not None and dtype_out is not None:
        raise ValueError(
            "Cache pollution only available when max sizes are unspecified")

    if dtype_in is None:
        dtype_in = np.int32
    if dtype_out is None:
        dtype_out = dtype_in

    try:
        max_shape_bytes = min(queue.device.max_mem_alloc_size, queue.device.global_variable_preferred_total_size // 2)
        if max_shape_bytes == 0:
            raise cl._cl.LogicError
    except cl._cl.LogicError:
        if "A100-SXM4" in queue.device.name:
            max_shape_bytes = min(queue.device.max_mem_alloc_size, queue.device.global_mem_size // 5)
        else:
            max_shape_bytes = min(queue.device.max_mem_alloc_size, queue.device.global_mem_size // 2)

    #max_shape_bytes = (max_shape_bytes // dtype_in().itemsize)*dtype_in().itemsize

    if n_in_max is None:
        n_in_max = max_shape_bytes // max(dtype_in().itemsize, dtype_out().itemsize)
    if n_out_max is None:
        n_out_max = n_in_max

    n_in_max_bytes = n_in_max*dtype_in().itemsize
    n_out_max_bytes = n_out_max*dtype_out().itemsize

    assert n_in_max_bytes <= queue.device.max_mem_alloc_size
    assert n_out_max_bytes <= queue.device.max_mem_alloc_size
    assert n_in_max_bytes + n_out_max_bytes <= queue.device.global_mem_size

    pollute_size = n_in_max_bytes
    # Could probably get by with something smaller
    #pollute_size = min(100*queue.device.global_mem_cache_size,
    #                    n_in_max_bytes)

    ogti = n_out_max > n_in_max
    igto = n_in_max > n_out_max
    out_in_ratio = n_out_max / n_in_max
    in_out_ratio = n_in_max / n_out_max

    n_max = max(n_in_max, n_out_max)
    read_index = "j,i % n_in" if ogti else "j,i"
    write_index = "j,i % n_out" if igto else "j,i"

    knl = lp.make_kernel(
        "{[i,j]: 0<=i<ni and 0<=j<nj}",
        f"""
        output[{read_index}] = input[{write_index}]
        """,
        assumptions="ni>=0 and nj>=0",
    )
    knl = lp.add_dtypes(knl, {"output": dtype_out, "input": dtype_in})
    knl = lp.set_options(knl, "no_numpy")
    knl_orig = knl.copy()

    # Just do this once so don't need to do in the tuning loop
    d_in_buf, d_out_buf = get_buffers(
        queue, dtype_in, n_in_max, dtype_out=dtype_out,
        n_dtype_out=n_out_max, fill_on_device=True)

    results_dict = {}

    # Probably excessive searching for most purposes
    nj_range = [4] if fast else range(1, 9)
    local_size_range = [128] if fast else 32*np.array([1, 2, 4, 8, 16, 32])

    for nj in nj_range:
        for local_size in local_size_range:

            word_count_list = get_word_counts(n_max, minimum=nj)

            for n in word_count_list:
                knl = knl_orig

                events = []
                ni = n // nj
                knl = lp.fix_parameters(knl, ni=ni)
                knl = lp.fix_parameters(knl, nj=nj)
                if ogti:
                    n_out = ni
                    n_in = int(np.ceil(in_out_ratio*n_out))
                    knl = lp.fix_parameters(knl, n_out=n_out)
                elif igto:
                    n_in = ni
                    n_out = int(np.ceil(out_in_ratio*n_in))
                    knl = lp.fix_parameters(knl, n_in=n_in)
                else:
                    n_in = ni
                    n_out = ni

                end_slab = 0 if ni % min(local_size, ni) == 0 else 1
                knl = lp.split_iname(
                    knl, "i", min(
                        local_size, ni), inner_tag="l.0", outer_tag="g.0", slabs=(
                        0, end_slab))
                knl = lp.tag_inames(knl, [("j", "unr")])

                inpt = cl.array.Array(queue, (nj, n_in), dtype_in, data=d_in_buf)
                outpt = cl.array.Array(queue, (nj, n_out), dtype_out, data=d_out_buf)

                dt_avg = 0
                dt_max = 0
                dt_min = np.inf
                events = []

                for j in range(5):
                    if pollute_caches:
                        cl.enqueue_copy(queue, d_out_buf, d_in_buf,
                                        byte_count=pollute_size)
                    knl(queue, input=inpt, output=outpt)
                for j in range(ntrials):
                    if pollute_caches:
                        cl.enqueue_copy(queue, d_out_buf, d_in_buf,
                                        byte_count=pollute_size)
                    evt, _ = knl(queue, input=inpt, output=outpt)
                    events.append(evt)

                cl.wait_for_events(events)
                for evt in events:
                    dt = evt.profile.end - evt.profile.start
                    dt_avg += dt
                    if dt > dt_max:
                        dt_max = dt
                    if dt < dt_min:
                        dt_min = dt

                # Convert to seconds
                dt_avg = dt_avg / ntrials / 1e9
                dt_max = dt_max / 1e9
                dt_min = dt_min / 1e9

                # Calculate bandwidth in GBps
                nbytes_transferred = dtype_in().itemsize*np.product(inpt.shape) + \
                                        dtype_out().itemsize*np.product(outpt.shape)
                result = BandwidthTestResult(str(queue.device), dt_avg,
                            dt_min, dt_max, nbytes_transferred, "loopy",
                            (nj, local_size))

                # Keep the result with the lowest tmin
                if nbytes_transferred not in results_dict:
                    results_dict[nbytes_transferred] = result
                elif result.tmin < results_dict[nbytes_transferred].tmin:
                    results_dict[nbytes_transferred] = result

                if print_results:
                    avg_bw = nbytes_transferred/dt_avg/1e9
                    max_bw = nbytes_transferred/dt_min/1e9
                    min_bw = nbytes_transferred/dt_max/1e9

                    print(
                        f"Bytes: {nbytes_transferred}, \
                         Avg time: {dt_avg}, \
                         Min time: {dt_min}, \
                         Max time: {dt_max}, \
                         Avg GBps: {avg_bw}, \
                         Max GBps: {max_bw}, \
                         Min GBps  {min_bw}")

                # Need to have read access on both input and output arrays
                # for this to work

                #from pyopencl.array import sum as clsum
                #if n_in == n_out:
                #    diff = (inpt - outpt)
                #    if  clsum(inpt - outpt) != 0:
                #        print("INCORRECT COPY")

    def key(result):
        return result.bytes_transferred
    return tuple(sorted(results_dict.values(), key=key))


def enqueue_copy_bandwidth_test_with_queues_like(
        queue, dtype=None, fill_on_device=True, max_used_bytes=None):

    queues = get_queues_like(queue)

    return tuple([enqueue_copy_bandwidth_test(q, dtype=dtype,
                    fill_on_device=fill_on_device,
                    max_used_bytes=max_used_bytes) for q in queues])


def enqueue_copy_bandwidth_test(
        queue, dtype=None, fill_on_device=True, max_used_bytes=None,
        ntrials=100, print_results=False, pollute_caches=True):

    if pollute_caches and max_used_bytes is not None:
        raise ValueError(
            "Cache pollution only available when max_used_bytes is unspecified")

    if dtype is None:
        dtype = np.int32 if fill_on_device else np.int8

    try:
        max_shape_bytes = min(queue.device.max_mem_alloc_size, queue.device.global_variable_preferred_total_size // 2)
        if max_shape_bytes == 0:
            raise cl._cl.LogicError
    except cl._cl.LogicError:
        max_shape_bytes = min(queue.device.max_mem_alloc_size, queue.device.global_mem_size // 2)

    if max_used_bytes is not None:
        assert max_used_bytes // 2 <= queue.device.max_mem_alloc_size
        assert max_used_bytes <= queue.device.global_mem_size
        max_shape_bytes = max_used_bytes // 2

    # Redefine max_shape_bytes in case there is a remainder in the division
    word_size = dtype().itemsize
    max_shape_dtype = max_shape_bytes // word_size
    max_shape_bytes = max_shape_dtype*word_size
    max_used_bytes = 2*max_shape_bytes

    d_in_buf, d_out_buf = get_buffers(
        queue, dtype, max_shape_dtype, fill_on_device=fill_on_device)

    word_count_list = get_word_counts(max_shape_dtype)
    results_list = []

    pollute_size = max_shape_bytes#queue.device.max_mem_alloc_size
    # Could probably get by with something smaller
    #pollute_size = min(100*queue.device.global_mem_cache_size,
    #                   max_shape_bytes)

    for word_count in word_count_list:
        dt_max = 0
        dt_min = np.inf
        dt_avg = 0

        events = []
        byte_count = word_size*word_count

        # Warmup
        for i in range(5):
            if pollute_caches:
                cl.enqueue_copy(queue, d_out_buf, d_in_buf, byte_count=pollute_size)
            evt = cl.enqueue_copy(queue, d_out_buf, d_in_buf, byte_count=byte_count)
        for i in range(ntrials):
            if pollute_caches:
                cl.enqueue_copy(queue, d_out_buf, d_in_buf, byte_count=pollute_size)
            evt = cl.enqueue_copy(queue, d_out_buf, d_in_buf, byte_count=byte_count)
            events.append(evt)

        cl.wait_for_events(events)
        for evt in events:
            dt = evt.profile.end - evt.profile.start
            dt_avg += dt
            if dt > dt_max:
                dt_max = dt
            if dt < dt_min:
                dt_min = dt

        # Convert to seconds

        dt_avg = dt_avg / ntrials / 1e9
        dt_max = dt_max / 1e9
        dt_min = dt_min / 1e9

        # Calculate bandwidth in GBps
        nbytes_transferred = 2*byte_count
        result = BandwidthTestResult(
            str(queue.device), dt_avg, dt_min, dt_max,
            nbytes_transferred, "enqueue_copy")
        results_list.append(result)

        if print_results:
            avg_bw = nbytes_transferred/dt_avg/1e9
            max_bw = nbytes_transferred/dt_min/1e9
            min_bw = nbytes_transferred/dt_max/1e9

            print(
                f"Bytes: {nbytes_transferred}, \
                 Avg time: {dt_avg}, \
                 Min time: {dt_min}, \
                 Max time: {dt_max}, \
                 Avg GBps: {avg_bw}, \
                 Max GBps: {max_bw}, \
                 Min GBps  {min_bw}")

    return tuple(results_list)


# For when the latency is already known
def get_beta_model(results_list, latency):

    M = np.array([(result.bytes_transferred, result.tmin)
                 for result in results_list])
    coeffs = np.linalg.lstsq(M[:, :1], M[:, 1] - latency, rcond=None)[0]

    return (coeffs[0])


# Returns latency in seconds and inverse bandwidth in seconds per byte
def get_alpha_beta_model(results_list, total_least_squares=False):

    # Could take the latency to be the lowest time ever seen,
    # but that might be limited by the precision of the event timer

    if total_least_squares:
        M = np.array([(1, result.bytes_transferred, result.tmin)
                     for result in results_list])
        U, S, VT = np.linalg.svd(M)
        coeffs = ((-1/VT[-1, -1])*VT[-1, :-1]).flatten()
    else:
        M = np.array([(1, result.bytes_transferred, result.tmin)
                     for result in results_list])
        coeffs = np.linalg.lstsq(M[:, :2], M[:, 2], rcond=None)[0]

    return (coeffs[0], coeffs[1],)


def plot_bandwidth(results_list):

    latency, inv_bandwidth = get_alpha_beta_model(results_list)
    print("LATENCY:", latency, "BANDWIDTH:", 1/inv_bandwidth/1e9)
    M = np.array([(result.bytes_transferred, result.max_bandwidth)
                 for result in results_list])

    best_fit_bandwidth = M[:, 0]/(latency + M[:, 0]*inv_bandwidth)/1e9

    plt.figure()
    plt.semilogx(M[:, 0], M[:, 1]/1e9)
    plt.semilogx(M[:, 0], best_fit_bandwidth)
    plt.xlabel("Bytes read + bytes written")
    plt.ylabel("Bandwidth (GBps)")
    plt.show()


def plot_split_alpha_beta(results_list):

    # Find index of the the highest local maximum
    # then find the index of the next local minimum
    # Use those as the indices

    highest_delta = 0
    for ind, result in enumerate(results_list):
        if ind > 0 and ind < len(results_list) - 1:
            if result.max_bandwidth > results_list[ind-1].max_bandwidth and \
                    result.max_bandwidth > results_list[ind+1].max_bandwidth:
                delta = abs(result.max_bandwidth -
                            results_list[ind-1].max_bandwidth) + \
                        abs(result.max_bandwidth -
                            results_list[ind+1].max_bandwidth)
                if delta > highest_delta:
                    highest_delta = delta
                    split_index_lower = ind + 1

    highest_delta = 0
    for ind, result in enumerate(results_list):
        if ind > 0 and ind < len(results_list) - 1 and ind >= split_index_lower:
            if result.max_bandwidth < results_list[ind-1].max_bandwidth and \
                    result.max_bandwidth < results_list[ind+1].max_bandwidth:
                delta = abs(result.max_bandwidth -
                            results_list[ind-1].max_bandwidth) + \
                        abs(result.max_bandwidth -
                            results_list[ind+1].max_bandwidth)
                if delta > highest_delta:
                    highest_delta = delta
                    split_index_upper = ind

    M = np.array([(result.bytes_transferred, result.max_bandwidth)
                 for result in results_list])

    latency, inv_bw = get_alpha_beta_model(results_list)
    s1_latency, s1_inv_bw = get_alpha_beta_model(results_list[:split_index_lower])
    s2_inv_bw = get_beta_model(results_list[split_index_upper:], latency)

    print("LATENCY:", latency, "BANDWIDTH:", 1/inv_bw/1e9)
    print("REGION 1 LATENCY:", s1_latency, "REGION 1 BANDWIDTH:", 1/s1_inv_bw/1e9)
    print("LATENCY:", latency, "REGION 2 BANDWIDTH:", 1/s2_inv_bw/1e9)

    best_fit_bw = M[:, 0]/(latency + M[:, 0]*inv_bw)/1e9
    s1_bw = M[:, 0]/(s1_latency + M[:, 0]*s1_inv_bw)/1e9
    s2_bw = M[:, 0]/(latency + M[:, 0]*s2_inv_bw)/1e9

    plt.figure()
    plt.semilogx(M[:, 0], M[:, 1]/1e9)
    plt.semilogx(M[:, 0], best_fit_bw)
    plt.semilogx(M[:split_index_lower, 0], s1_bw[:split_index_lower])
    plt.semilogx(M[split_index_upper:, 0], s2_bw[split_index_upper:])
    plt.xlabel("Bytes read + bytes written")
    plt.ylabel("Bandwidth (GBps)")
    plt.show()


def get_indices_from_queue(queue):
    dev = queue.device
    if "NVIDIA" in dev.vendor:
        pcie_id = dev.pci_bus_id_nv
    elif dev.vendor == "Advanced Micro Devices":
        pcie_id = dev.pcie_id_amd
    else:
        raise RuntimeError("Device does not have a PCI-Express bus ID")

    for platform_number, platform in enumerate(cl.get_platforms()):
        for device_number, d in enumerate(platform.get_devices()):
            if "NVIDIA" in d.vendor:
                d_pcie_id = d.pci_bus_id_nv
            elif dev.vendor == "Advanced Micro Devices":
                d_pcie_id = d.pcie_id_amd
            else:
                d_pcie_id = None

            if pcie_id == d_pcie_id:
                # We found the device
                return platform_number, device_number

    raise ValueError("Unable to obtain platform and device numbers from queue")


def get_max_bandwidth_clpeak(queue=None, platform_number=0, device_number=0):
    
    if queue is not None:
        platform_number, device_number = get_indices_from_queue(queue)

    output = check_output(shlex.split(f"clpeak -p {platform_number} -d {device_number} --global-bandwidth"))
    output_split = output.decode().split()
    bandwidths = []
    for ind, entry in enumerate(output_split):
        if "float" in entry:
            bandwidths.append(float(output_split[ind + 2]))
    max_el = np.array(bandwidths).max()*1e9
    return max_el


def get_max_flop_rate_clpeak(dtype, queue=None, platform_number=0, device_number=0):
    
    if queue is not None:
        platform_number, device_number = get_indices_from_queue(queue)

    if dtype == np.float64:
        float_str = "dp"
    elif dtype == np.float32:
        float_str = "sp"
    elif dtype == np.float16:
        float_str = "hp"
    else:
        raise ValueError(f"Cannot handle dtype {dtype}")

    output = check_output(shlex.split(f"clpeak -p {platform_number} -d {device_number} --compute-{float_str}"))

    output_split = output.decode().split()
    flop_rates = []
    for ind, entry in enumerate(output_split):
        # Work around for message about half precision support in dp test
        if (dtype == np.float64 and "double" in entry) or \
            (dtype == np.float32 and "float" in entry) or \
            (dtype == np.float16 and "half" in entry):
            if not output_split[ind + 2] == "support":
                flop_rates.append(float(output_split[ind + 2]))
    if len(flop_rates) == 0:
        raise ValueError(f"No support for {dtype} on device")

    max_el = np.array(flop_rates).max()*1e9
    return max_el

# Remove the effect of latency from the bandwidth calculation.
def calc_latency_adjusted_bandwidth(latency, total_time, bytes_transferred):
    denominator = total_time - latency
    assert latency >= 0
    assert denominator > 0
    return bytes_transferred / denominator

# Adds the effect of latency to the bandwidth calculation
def calc_effective_bandwidth(latency, bandwidth, bytes_transferred):
    return bytes_transferred / (bytes_transferred / bandwidth + latency)

def get_min_device_memory_latency(results_list):
    sorted_list = sorted(results_list, reverse=False,
                         key=lambda result: result.tmin)
    return sorted_list[0].tmin
   
def get_max_device_memory_bandwidth(results_list):
    sorted_list = sorted(results_list, reverse=True,
                         key=lambda result: result.max_bandwidth)
    return sorted_list[0].max_bandwidth
    
def get_latency_adjusted_max_device_memory_bandwidth(results_list):
    sorted_list = sorted(results_list, reverse=False,
                         key=lambda result: result.tmin)
    latency = sorted_list[0].tmin

    sorted_list = sorted(results_list, reverse=True,
                         key=lambda result: result.max_bandwidth)
    tmin = sorted_list[0].tmin
    bytes_transferred = sorted_list[0].bytes_transferred
    return calc_latency_adjusted_bandwidth(latency, tmin, bytes_transferred)
    

if __name__ == "__main__":

    context = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(
        context, properties=cl.command_queue_properties.PROFILING_ENABLE)


    loopy_results_list = loopy_bandwidth_test(queue, fast=True,
        print_results=True, fill_on_device=True)
    enqueue_results_list = enqueue_copy_bandwidth_test(
        queue, dtype=None, fill_on_device=False, max_used_bytes=None,
        print_results=True)

    #plot_split_alpha_beta(loopy_results_list)
    #plot_split_alpha_beta(enqueue_results_list)
    #exit()
    combined_list = loopy_results_list + enqueue_results_list

    clpeak_bw = get_max_bandwidth_clpeak(queue=queue, platform_number=0, device_number=0)
    clpeak_flop_rate = get_max_flop_rate_clpeak(np.float64, queue=queue, platform_number=0, device_number=0)

    flop_rate = get_theoretical_maximum_flop_rate(queue, np.float64)
    print("MAX THEORETICAL FLOP RATE (GFLOP/s)", flop_rate / 1e9)
    print("clpeak MAX FLOP RATE (GLOP/s)", clpeak_flop_rate / 1e9)
    print("clpeak MAX BW (GB/s)", clpeak_bw/1e9)
    print("Device memory latency (s)", get_min_device_memory_latency(combined_list))
    print("Loopy kernel MAX BW (GB/s)", get_max_device_memory_bandwidth(combined_list)/1e9)
    print("Latency adjusted MAX BW (GB/s)", get_latency_adjusted_max_device_memory_bandwidth(combined_list)/1e9)
    exit()



    sorted_list = sorted(loopy_results_list, reverse=True,
                         key=lambda result: result.avg_bandwidth)
    print(sorted_list[0])
    # Loopy kernel is probably more indicative of real world performance
    #plot_bandwidth(loopy_results_list)
    #plot_split_alpha_beta(enqueue_results_list)
    #plot_split_alpha_beta(loopy_results_list)

    """
    tmin_key = lambda result: result.tmin

    #results_list_list_enqueue = enqueue_copy_bandwidth_test_with_queues_like(queue)
    #combined_list_enqueue = [sorted(tup, key=tmin_key)[0]
    #    for tup in zip(*results_list_list_enqueue)]

    # Can't use PoCL until https://github.com/pocl/pocl/pull/1094 is fixed
    #results_list_list_loopy = loopy_bandwidth_test_with_queues_like(queue,
    #   fast=True)
    #combined_list_loopy = [sorted(tup, key=tmin_key)[0]
    #    for tup in zip(*results_list_list_loopy)]

    #combined_list = [*combined_list_loopy, *combined_list_enqueue]

    # Eliminate redundant data points, save the fastest minimum time
    results_dict = {}
    for entry in combined_list:
        nbytes_transferred = entry.bytes_transferred
        if nbytes_transferred not in results_dict:
            results_dict[nbytes_transferred] = entry
        elif entry.tmin < results_dict[nbytes_transferred].tmin:
            results_dict[nbytes_transferred] = entry

    bytes_key = lambda result: result.bytes_transferred
    combined_list = sorted(results_dict.values(), key=bytes_key)
    for entry in combined_list:
        print(entry.bytes_transferred, entry.tmin, entry.max_bandwidth)

    print()
    print("Loopy results:", get_alpha_beta_model(loopy_results_list))
    print("Enqueue copy results:", get_alpha_beta_model(enqueue_results_list))

    plot_bandwidth(combined_list)
    """
