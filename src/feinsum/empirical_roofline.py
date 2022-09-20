import pyopencl as cl
import numpy as np
import pyopencl.array as clarray
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
                queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
                queues.append(queue)
    return queues

@dataclass
class BandwidthTestResult():

    device: str
    tavg: float
    tmin: float
    tmax: float
    bytes_transferred: int
    #nbytes_read: int = None
    #nbytes_written: int = None

    @property
    def avg_bandwidth(self):
        return self.bytes_transferred/self.tavg

    @property
    def max_bandwidth(self):
        return self.bytes_transferred/self.tmin

    @property
    def min_bandwidth(self):
        return self.bytes_transferred/self.tmax
    
def enqueue_copy_bandwidth_test_with_queues_like(queue, dtype=None, fill_on_device=True, max_used_bytes=None):

    queues = get_queues_like(queue)

    return tuple([enqueue_copy_bandwidth_test(q, dtype=dtype,
                    fill_on_device=fill_on_device,
                    max_used_bytes=max_used_bytes) for q in queues])

def get_buffers(queue, dtype, fill_on_device, max_shape_dtype):

    max_shape_bytes = max_shape_dtype*dtype().itemsize
    context = queue.context

    d_out_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.HOST_NO_ACCESS, size=max_shape_bytes)

    if fill_on_device: # Requires making a READ_WRITE buffer instead of a READ_ONLY buffer
        if dtype in {np.float64, np.float32, np.int32, np.int64}:
            allocator = ImmediateAllocator(queue)
            d_in_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS, size=max_shape_bytes)
            d_in_buf_arr = cl.array.Array(queue, (max_shape_dtype,), dtype, allocator=allocator, data=d_in_buf)
            clrandom.fill_rand(d_in_buf_arr, queue=queue)
        else:
            raise ValueError(f"Cannot fill array with {dtype} on the device")
    else:

        from psutil import virtual_memory
        
        if np.issubdtype(dtype, np.integer):
            if virtual_memory().available < max_shape_bytes:
                raise ValueError("Not enough host memory to fill the buffer from the host")

            max_val = np.iinfo(dtype).max
            min_val = np.iinfo(dtype).min

            # Does the randint do any internal conversions?
            h_in_buf = np.random.randint(min_val, high=max_val + 1, size=max_shape_dtype, dtype=dtype)
        elif np.issubdtype(dtype, np.float):
            # The host array is formed as a float64 before being copied and converted
            if virtual_memory().available < max_shape_dtype*(np.float64().itemsize + word_size):
                raise ValueError("Not enough host memory to fill the buffer from the host")
            h_in_buf = np.random.rand(max_shape_dtype).astype(dtype)
        else:
            raise ValueError("Unsupported dtype: {dtype}")

        d_in_buf = cl.Buffer(context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.HOST_NO_ACCESS | cl.mem_flags.COPY_HOST_PTR,
            size=max_shape_bytes, hostbuf=h_in_buf)

        
        #TODO: Copy small chunks at a time if the array size is large.
        # Is this actually needed? This will require HOST_WRITE_ONLY flags
 
        #d_in_buf = cl.Buffer(context,
        #    cl.mem_flags.READ_ONLY | cl.mem_flags.HOST_WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
        #    size=max_shape_bytes, hostbuf=h_in_buf)
           
        #evt = cl.enqueue_copy(queue, d_in_buf, h_in_buf)

    return d_in_buf, d_out_buf

def enqueue_copy_bandwidth_test(queue, dtype=None, fill_on_device=True, max_used_bytes=None, ntrials=1000):

    if dtype is None:
        dtype = np.int32 if fill_on_device else np.int8

    word_size = dtype().itemsize

    if max_used_bytes is None:
        max_shape_bytes = queue.device.max_mem_alloc_size
    else:
        if max_used_bytes <= queue.device.max_mem_alloc_size:
            max_shape_bytes = max_used_bytes // 2
        else:
            raise ValueError("max_used_bytes is greater than the available device memory")

    max_shape_dtype = max_shape_bytes // word_size
    # Redefine max_shape_bytes in case there is a remainder in the division
    max_shape_bytes = max_shape_dtype*word_size
    max_used_bytes = 2*max_shape_bytes

    d_in_buf, d_out_buf = get_buffers(queue, dtype, fill_on_device, max_shape_dtype)

    len_list = []

    word_count = 1
    # Get some non-multiples of two
    while word_count <= max_shape_dtype:
        len_list.append(int(np.floor(word_count)))
        word_count = word_count*1.5
    # Get multiples of two
    for i in range(0,int(np.floor(np.log2(max_shape_dtype)) + 1)):
        len_list.append(2**i)
    len_list = sorted(list(set(len_list)))

    results_list = []
    
    for word_count in len_list:
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
        dt_avg  = dt_avg / ntrials / 1e9
        dt_max = dt_max / 1e9
        dt_min = dt_min / 1e9

        # Calculate bandwidth in GBps
        nbytes_transferred = 2*byte_count
        avg_bw = nbytes_transferred/dt_avg/1e9
        max_bw = nbytes_transferred/dt_min/1e9
        min_bw = nbytes_transferred/dt_max/1e9

        result = BandwidthTestResult(str(queue.device), dt_avg, dt_min, dt_max, nbytes_transferred)
        results_list.append(result)

        print(f"{nbytes_transferred} {dt_avg} {dt_min} {dt_max} {avg_bw} {max_bw} {min_bw}")

    return tuple(results_list)

# Returns latency in seconds and inverse bandwidth in seconds per byte
def get_alpha_beta_model(results_list, total_least_squares=False):

    # Could take the latency to be the lowest time ever seen,
    # but that might be limited by the precision of the event timer

    if total_least_squares:
        M = np.array([(1, result.bytes_transferred, result.tmin) for result in results_list])
        U, S, VT = np.linalg.svd(M)
        coeffs = ((-1/VT[-1,-1])*VT[-1,:-1]).flatten()
    else:
        M = np.array([(1, result.bytes_transferred, result.tmin) for result in results_list])
        coeffs = np.linalg.lstsq(M[:,:2], M[:,2], rcond=None)[0]

    return (coeffs[0], coeffs[1],)

def plot_bandwidth(results_list):
    import matplotlib.pyplot as plt

    latency, inv_bandwidth = get_alpha_beta_model(results_list)
    print("LATENCY:", latency, "BANDWIDTH:", 1/inv_bandwidth/1e9)
    M = np.array([(result.bytes_transferred, result.max_bandwidth) for result in results_list])

    best_fit_bandwidth = M[:,0]/(latency + M[:,0]*inv_bandwidth)/1e9
    
    fig = plt.figure()
    plt.semilogx(M[:,0], M[:,1]/1e9)
    plt.semilogx(M[:,0], best_fit_bandwidth)
    plt.xlabel("Bytes read + bytes written")
    plt.ylabel("Bandwidth (GBps)")
    plt.show()

if __name__ == "__main__":
    
    context = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)


    #results_list = enqueue_copy_bandwidth_test(queue, dtype=None, fill_on_device=True, max_used_bytes=None)

    #get_alpha_beta_model(results_list)
    #plot_bandwidth(results_list)

    results_list_list = enqueue_copy_bandwidth_test_with_queues_like(queue, max_used_bytes=None)
    
    key = lambda result: result.tmin
    combined_list = [sorted(tup, key=key)[0] for tup in zip(*results_list_list)]

    for results_list in results_list_list:
        coeffs = get_alpha_beta_model(results_list)
        print(coeffs)

    plot_bandwidth(combined_list)

