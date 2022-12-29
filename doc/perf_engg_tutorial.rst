.. _tutorial-1:

Tutorial: Expressing code-transformation space
==============================================

We provide an example of code-transformation space for batched matrix-vector
multiplication kernels as:

.. literalinclude:: ../src/feinsum/tuning/impls/demo_transform_space.py
   :start-after: BEGINEXAMPLE
   :end-before: ENDEXAMPLE


The main parts of the script include:

- ``transform`` function that implements the parametric transformations.
- ``einsum_arg`` decorator that parametrizes the sub-class of einstein
  summations it targets.
- ``transform_param`` decorator that prescribes the bounds of the transform
  space.
- ``match_t_unit_to_einsum`` re-interprets the relevant expresions in the
  translation unit as per the provided reference batched einsum and provides
  a substitution mapping from the einsum's entities to the kernel's
  entities as:

  * An einsum's index corresponds to one of the inames in the translation unit
  * An einsum's argument corresponds to one the substitution rules in the
    translation unit

We ask :mod:`feinsum` to record facts about the transform space as:

.. code-block:: python

    import pyopencl as cl

    cl_ctx = cl.create_some_context()

    my_einsum = f.einsum("xik,kj->xij",
                         f.array((np.inf, 35, 35), np.float64),
                         f.array((np.inf, 35, 35), np.float64))
    f.autotune(my_einsum, cl_ctx=cl_ctx, module_path=__file__)


Using the above script, kernel for *my_einsum* is tuned with the previously
defined transform-space. *feinsum* leverages `OpenTuner
<https://opentuner.org/>`__ to traverse the search space.

On running the code we should see messages printed to ``stderr`` as:

.. code-block::
  
    $ python main.py

    [     2s]    INFO feinsum.tuning: {'n_x_per_wg': 26, 'n_wi_per_x': 12}
    [     3s]    INFO feinsum.measure: Statistically verified the soundness of the transformation
    [     6s]    INFO feinsum.tuning: 
    ╒═════════╤═════════════════╤═════════════════╕
    │ Dtype   │ Measured GOps/s │ Roofline GOps/s │
    ├─────────┼─────────────────┼─────────────────┤
    │ float64 │ 22.3            │ 42.1            │
    ╘═════════╧═════════════════╧═════════════════╛
    [     8s]    INFO feinsum.tuning: {'n_x_per_wg': 8, 'n_wi_per_x': 19}
    [     9s]    INFO feinsum.measure: Statistically verified the soundness of the transformation
    [    11s]    INFO feinsum.tuning: 
    ╒═════════╤═════════════════╤═════════════════╕
    │ Dtype   │ Measured GOps/s │ Roofline GOps/s │
    ├─────────┼─────────────────┼─────────────────┤
    │ float64 │ 29.3            │ 42.1            │
    ╘═════════╧═════════════════╧═════════════════╛
    [    13s]    INFO feinsum.tuning: {'n_x_per_wg': 10, 'n_wi_per_x': 26}
    [    14s]    INFO feinsum.measure: Statistically verified the soundness of the transformation
    [    17s]    INFO feinsum.tuning: 
    ╒═════════╤═════════════════╤═════════════════╕
    │ Dtype   │ Measured GOps/s │ Roofline GOps/s │
    ├─────────┼─────────────────┼─────────────────┤
    │ float64 │ 19.3            │ 42.1            │
    ╘═════════╧═════════════════╧═════════════════╛

