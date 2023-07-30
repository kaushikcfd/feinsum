.. _dsgn:

Design
======

.. _dsgn_explain_deps:

Dependencies
------------

*Feinsum*'s design is closely tied with :mod:`loopy` and OpenCL [OpenCL_2011]_.
Loopy provides a language for specifying code-transformations based on the
Polyhedral model and OpenCL-specification to support various platforms (CPUs,
GPUs, DSPs, etc.) at the same time keeping *feinsum*'s code maintainable by
avoiding separate code-paths for CUDA, ISPC, etc. with virtually no performance
gains. (see `POCL <http://portablecl.org/>`_)


.. _dsgn_batched_einsum_defn:

What is a “Batched Einstein Summation”?
---------------------------------------

Batched Einstein Summations (einsum) represents a computational primitive to
serve as an abstraction between a *performance engineer* and a *compiler*. The
compiler could lower the program as a composition of certain computational
primitives and enjoy the benefits of code-transformations on those primitives
prescribed by a performance engineers. Some examples of such computational
primitives for similar purposes are: `BLAS <https://netlib.org/blas/>`_,
`LAPACK <https://netlib.org/lapack/>`_, `DNN
<https://docs.nvidia.com/deeplearning/cudnn/api/index.html>`_, etc.

**Definition**: A Batched Einstein Summation (:math:`\mathcal{B}_m^k`) describes a computation
with :math:`k` multi-dimensional arrays :math:`A_1, A_2 \ldots, A_k` as the
inputs and :math:`m`-multi-dimensional arrays :math:`R_1, R_2, \ldots, R_m` as
the outputs. The outputs are the results of an Einstein summation operation
with :math:`n` operands. It is defined by a 4-tuple :math:`(\mathcal{E},
\mathcal{A}_{m \times n}, \mathcal{T}, \mathcal{S})`:

#. :math:`\mathcal{E}` is the indexing expression used in defining an Einstein summation
   with :math:`n` multi-dimensional arrays as the operands. We borrow the
   grammar of :math:`\mathcal{E}` from ``numpy.einsum``. Each :math:`R_1,
   \ldots R_m` is a result of an E -einstein summation with :math:`n` operands.

#. :math:`\mathcal{A}_{m \times n}` is the argument matrix where each
   :math:`\mathcal{A}_{i j} \subseteq \{ A_{1,} \ldots, A_k \}` is a set of
   array names. The :math:`j`-th operand of the einstein summation computing
   :math:`R_i` is the term :math:`\prod_{a \in \mathcal{A}_{i j}}
   a [\mathcal{E}_{j, 1}, \mathcal{E}_{j, 2}, \ldots]` where,
   :math:`\mathcal{E}_{j, 1}, \mathcal{E}_{j, 2}, \ldots` are the indices for the
   :math:`j`-th operand of :math:`\mathcal{E}`.

#. :math:`\mathcal{T}` is a mapping from the input arrays to their
   numeric data types in the computation.

#. :math:`\mathcal{S}` is a mapping from the input arrays to their
   shapes. A shape is a tuple of non-negative integers or :math:`\infty`.
   :math:`\infty` is a symbol for a non-negative integer which is much greater
   than the cache-size of the target hardware.

No two arrays in the set :math:`\{ A_1, A_{2,} \ldots ., A_k, R_1, R_2, \ldots
., R_m \}` alias each other.


In :mod:`feinsum`, we use a slightly modified definition of *Batched-einsum* to
be able to be able to apply the transformation knowledge to a wider-class of
expressions without a noticeable increase in complexity on the performance
engineer's behalf. The transformations are recorded on
:mod:`loopy` kernels such that the :math:`j`-th operand of the einsum computing
:math:`R_i` is of the form :math:`\rho_{ij}(a_1[\mathcal{E}_{j, 1},
\mathcal{E}_{j, 2}, \ldots], a_2[\mathcal{E}_{j, 1}, \mathcal{E}_{j, 2},
\ldots], \ldots)`, where, :math:`\mathcal{A}_{i j} = \{a_1, a_2, \ldots\}` and
:math:`\rho_{ij}` is a function of the form
:math:`\rho_{ij}:\mathcal{T}[a_1]\times\mathcal{T}[a_2]\times\ldots\mapsto
\mathcal{C}\left(\mathcal{T}[a_1]\times\mathcal{T}[a_2]\times\ldots\right)`,
where, :math:`\mathcal{C}` is a mapping on the numeric data-types as provided
by :func:`numpy.result_type`.


The transformation writer prescribes the transformation space without making
any assumptions about :math:`rho_{ij}` i.e. the prescribed transformation space
has to be only a function of :math:`(\mathcal{E}, \mathcal{A}_{m \times n},
\mathcal{T}, \mathcal{S})` and NOT :math:`\rho_{m \times n}`.

One might note that the Batched-einsum expressions are a sub-class of the
modified Batched-einsum expressions by simply using :math:`\rho_{ij}(x_1, x_2,
\ldots) = x_1\cdot x_2\cdot\ldots`. Throughout this manual we will use
Batched-einsum and modified Batched-einsum interchangeably.

.. _dsgn_loopy_grammar:

Grammar of :mod:`loopy` kernels
-------------------------------

We rely on a grammar of :mod:`loopy` that defines a Batched-einsum expression
to serve as a contract between the transformation implementer and the
optimizing compiler that would reuse the transformations. We also point out
that a grammar of loopy kernels indirectly also specifies a *schedule* as per
the code-generation implementation in :mod:`loopy`.

We first describe the grammar of a loopy kernel that corresponds to a Batched
einsum kernel with a single output. Extending this grammar to a Batched-einsum
with more than one outputs is trivial and is covered via examples.

#. The einsum expression is matched to a single instruction inside the kernel.
#. The instruction must contain at most one sum-reduction expression in its
   RHS. This includes the predicates guarding the instruction's execution.
#. The multiplicative terms within the expression over which sum-reduction
   is being performed are inferred as the operand expression of the einsum.
#. The loopy kernel's domain formed by the loops nesting outside the
   instruction along with the reduction inames' domains must correspond to
   that of a hypercube.
#. The #dimensions of the instruction's assignee array must be equal to
   the loop nest dimension surrounding the instruction.
#. The instruction must write to an array (in the global address space)
   such that all the iname dependencies of the instruction must appear
   in the indexing expression of the assignee array of the instruction.
#. Any substitution invocation in the instruction's RHS is considered as
   an argument of the einsum expression.
#. Similar to the constraints on indexing into the assignee array the
   arguments to the substitution invocations must be just inames (either
   reduction inames or the inames corresponding to the instruction's loop
   nest).
#. The einsum is constructed as:
    - The indexing expression :math:`\mathcal{E}`'s output indices are obtained
      by reading the assignee array's indexing inames.
    - The indexing expression :math:`\mathcal{E}`'s input operands' indices are obtained
      by gathering the substitution rule invocation's arguments.
    - The numeric data-type of a substitution rule is inferred by calling
      :func:`loopy.infer_unknown_types` on the substitution rule's expression.
    - The shapes of the input operands are inferred from the loopy kernel's domains.

With these rules we can infer an einsum expression from a :mod:`loopy` kernel.
Inferring a batched einsum expression is simply applying the above rules to
a collection of instructions in a loopy kernel. We rely on a canonicalization
routine (see :func:`~feinsum.canonicalization.canonicalize_einsum`) to ensure that the parsing
for the above grammar of expressions is deterministic.

We note that :mod:`feinsum` does not impose any constraints on the substitution
rule's expression. We leave that upon the compiler to ensure that the rule's RHS
has memory access pattern close to that of a multi-dimensional array with similar
stride patterns.

.. _dsgn_why_perf_engg:

Why keep a performance engineer in the loop?
--------------------------------------------

As of this writing, limited solutions are available that unify the heuristics
needed to generate roofline-performing code for a single Einstein summation.
Approaches that rely on an auto-tuning phase have been proposed for generating
optimized device codes for a sub-class of Einstein-summations and for
a particular architecture, for eg. Cogent [Kim_2019]_ generates optimized GPU
kernels for tensor-contractions.

Hence, *feinsum* provides abstractions to develop (and implement)
code-transformations for certain sub-classes of Batched-einsums.
