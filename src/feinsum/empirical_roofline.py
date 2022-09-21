import pyopencl as cl
import numpy as np
import loopy as lp
import pyopencl.clrandom as clrandom
from pyopencl.tools import ImmediateAllocator
from dataclasses import dataclass

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
                        ntrials=100, fast=True):

    if dtype_in is None:
        dtype_in = np.int32
    if dtype_out is None:
        dtype_out = dtype_in

    if n_in_max is None:
        n_in_max = queue.device.max_mem_alloc_size // dtype_in().itemsize
    if n_out_max is None:
        n_out_max = n_in_max

    n_in_max_bytes = n_in_max*dtype_in().itemsize
    n_out_max_bytes = n_out_max*dtype_out().itemsize

    if n_in_max_bytes > queue.device.max_mem_alloc_size:
        raise ValueError("Maximum input length exceeds maximum allocation size")
    if n_out_max_bytes > queue.device.max_mem_alloc_size:
        raise ValueError("Maximum output length exceeds maximum allocation size")

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
    knl = lp.set_options(knl, "no_numpy")  # Output code before editing it
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

                local_size = 128  # 256 or 512 seems to do the best
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

                for j in range(2):
                    knl(queue, input=inpt, output=outpt)
                for j in range(ntrials):
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
                avg_bw = nbytes_transferred/dt_avg/1e9
                max_bw = nbytes_transferred/dt_min/1e9
                min_bw = nbytes_transferred/dt_max/1e9

                result = BandwidthTestResult(str(queue.device), dt_avg,
                            dt_min, dt_max, nbytes_transferred, "loopy",
                            (nj, local_size))

                # Keep the result with the lowest tmin
                if nbytes_transferred not in results_dict:
                    results_dict[nbytes_transferred] = result
                elif result.tmin < results_dict[nbytes_transferred].tmin:
                    results_dict[nbytes_transferred] = result

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
        queue, dtype=None, fill_on_device=True, max_used_bytes=None, ntrials=1000):

    if dtype is None:
        dtype = np.int32 if fill_on_device else np.int8

    if max_used_bytes is None:
        max_shape_bytes = queue.device.max_mem_alloc_size
    else:
        max_shape_bytes = max_used_bytes // 2

    word_size = dtype().itemsize
    max_shape_dtype = max_shape_bytes // word_size
    # Redefine max_shape_bytes in case there is a remainder in the division
    max_shape_bytes = max_shape_dtype*word_size
    max_used_bytes = 2*max_shape_bytes

    if max_shape_bytes > queue.device.max_mem_alloc_size:
        raise ValueError("max_shape_bytes is larger than can be allocated")

    d_in_buf, d_out_buf = get_buffers(
        queue, dtype, max_shape_dtype, fill_on_device=fill_on_device)

    word_count_list = get_word_counts(max_shape_dtype)
    results_list = []

    for word_count in word_count_list:
        dt_max = 0
        dt_min = np.inf
        dt_avg = 0

        events = []
        byte_count = word_size*word_count

        # Warmup
        for i in range(5):
            evt = cl.enqueue_copy(queue, d_out_buf, d_in_buf, byte_count=byte_count)
        for i in range(ntrials):
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
        avg_bw = nbytes_transferred/dt_avg/1e9
        max_bw = nbytes_transferred/dt_min/1e9
        min_bw = nbytes_transferred/dt_max/1e9

        result = BandwidthTestResult(
            str(queue.device), dt_avg, dt_min, dt_max,
            nbytes_transferred, "enqueue_copy")
        results_list.append(result)

        print(
            f"Bytes: {nbytes_transferred}, \
             Avg time: {dt_avg}, \
             Min time: {dt_min}, \
             Max time: {dt_max}, \
             Avg GBps: {avg_bw}, \
             Max GBps: {max_bw}, \
             Min GBps  {min_bw}")

    return tuple(results_list)

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
    import matplotlib.pyplot as plt

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


if __name__ == "__main__":

    context = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(
        context, properties=cl.command_queue_properties.PROFILING_ENABLE)

    loopy_results_list = loopy_bandwidth_test(queue, fast=True)
    enqueue_results_list = enqueue_copy_bandwidth_test(
        queue, dtype=None, fill_on_device=True, max_used_bytes=None)

    combined_list = loopy_results_list + enqueue_results_list

    # Loopy kernel is probably more indicative of real world performance
    plot_bandwidth(loopy_results_list)

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
