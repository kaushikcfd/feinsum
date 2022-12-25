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
   space)
3. Defines (and implements) a specification to translate the recorded
   code-transformations to a subset of :mod:`loopy` kernels.

The rest of this manual is structured as follows. We outline our design
choices are elaborate in  Sec xxx. We provide a tutorial in Sec xx an Sec yy.
The API reference is generated in Sec. xxx.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    perf_engg_tutorial
    compiler_writer_tutorial
    design
    api
    refs

    🚀 Github <https://github.com/kaushikcfd/feinsum>
    💾 Download Releases <https://pypi.org/project/feinsum>
