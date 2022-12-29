=======
feinsum
=======

.. module:: feinsum

*Feinsum*, short for Fast-Einstein Summations, provides tools for managing
database of code-transformations on *Batched Einstein Summations*. Primarily,
it serves the following purposes:

1. Provides tools for performance engineers to record code-transformation
   spaces for a class of Batched Einstein Summations.
2. Automatically record facts (such as walltime, FLOPS) about the points in the
   transformation space specified by a performance engineer. (current
   implementation uses `OpenTuner <https://opentuner.org/>`_ to traverse the
   search space)
3. Defines (and implements) a specification to translate the recorded
   code-transformations to a subset of :mod:`loopy` kernels.

The rest of this manual is structured as follows. We outline our design choices
are elaborate in  Sec “:ref:`dsgn`\ ”. We provide tutorials in Sec
“:ref:`tutorial-1`\ ” an Sec “:ref:`tutorial-2`\ ”. The API reference is
generated in Sec. “:ref:`api`\ ”.

.. toctree::
    :maxdepth: 2
    :caption: Contents:
  
    install
    perf_engg_tutorial
    compiler_writer_tutorial
    design
    api
    refs

    🚀 Github <https://github.com/kaushikcfd/feinsum>
    💾 Download Releases <https://pypi.org/project/feinsum>
