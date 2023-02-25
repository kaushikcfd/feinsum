.. _tutorial-2:

Tutorial: Using the transformation knowledge
============================================

To transfer the knowledge present in the database a *feinsum* user provides
a :class:`~loopy.TranslationUnit` and which instructions in the translation
unit must be transformed. We show an example below.

Let us first create a kernel to apply transformations on:

.. testsetup::
    >>> import numpy as np
    >>> import loopy as lp
    >>> from feinsum.cl_utils import make_fake_cl_context
    >>> from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
    >>> cl_ctx = make_fake_cl_context("NVIDIA_TITAN_V")

.. doctest::

    >>> t_unit = lp.make_kernel(
    ...     ["{[iel_0,idof_0,jdof_0,x,r]: 0<=iel_0<Nel and 0<=idof_0,jdof_0<35 and 0<=x,r<3}",
    ...      "{[iel_1,idof_1,ifacedof,iface]: 0<=iel_1<Nel and 0<=idof_1<35 and 0<=ifacedof<15 and 0<=iface<4}"],
    ...     """
    ...     u_s(_0, _1)        := u[_0, _1]
    ...     f_0_s(_0, _1, _2)  := F_0[_0, _1, _2]
    ...     f_1_s(_0, _1, _2)  := F_1[_0, _1, _2]
    ...     f_2_s(_0, _1, _2)  := F_2[_0, _1, _2]
    ...     f_3_s(_0, _1, _2)  := F_3[_0, _1, _2]
    ...     jac_s(_0, _1, _2)  := J[_0, _1, _2]
    ...     D_s(_0, _1, _2)    := D[_0, _1, _2]
    ...     jac_face_s(_0, _1) := Jface[_0, _1]
    ...     L_s(_0, _1, _2)    := L[_0, _1, _2]
    ...
    ...     grad_out[x, iel_0, idof_0] = sum([jdof_0, r], \
    ...                                      jac_s(x, r, iel_0)*D_s(r, idof_0, jdof_0)*u_s(iel_0, jdof_0))
    ...
    ...     ... gbarrier {id=g_barrier_0, dep_query=(writes:grad_out)}
    ...
    ...     with {dep=g_barrier_0}
    ...         lift_0[iel_1, idof_1] = sum([iface, ifacedof], \
    ...                                     L_s(idof_1, iface, ifacedof)*jac_face_s(iface, iel_1)*f_0_s(iface, iel_1, ifacedof))
    ...         lift_1[iel_1, idof_1] = sum([iface, ifacedof], \
    ...                                     L_s(idof_1, iface, ifacedof)*jac_face_s(iface, iel_1)*f_1_s(iface, iel_1, ifacedof))
    ...         lift_2[iel_1, idof_1] = sum([iface, ifacedof], \
    ...                                     L_s(idof_1, iface, ifacedof)*jac_face_s(iface, iel_1)*f_2_s(iface, iel_1, ifacedof))
    ...         lift_3[iel_1, idof_1] = sum([iface, ifacedof], \
    ...                                     L_s(idof_1, iface, ifacedof)*jac_face_s(iface, iel_1)*f_3_s(iface, iel_1, ifacedof))
    ...     end
    ...     """)
    >>>
    >>> t_unit = lp.add_dtypes(t_unit, {"u": np.float64, "F_0": np.float64, "F_1": np.float64,
    ...                                 "F_2": np.float64, "F_3": np.float64, "J": np.float64,
    ...                                 "Jface": np.float64, "D": np.float64, "L": np.float64})


Now let's ask :mod:`feinsum` to parse the two batched einsums in the kernel according
to the specification (see ":ref:`dsgn_loopy_grammar`"):

.. doctest::

    >>> import feinsum as f
    >>> ensm1, _ = f.get_a_matched_einsum(t_unit, insn_match="writes:grad_out")
    >>> ensm2, _ = f.get_a_matched_einsum(t_unit, insn_match="writes:lift_*")


We now ask *feinsum* to query the available transformation knowledge and pick
the best one according to a user-relevant metric, which in this case is the
FLOP throughput.

.. doctest::
  
    >>> # Construct a 'cl_ctx' of type `pyopencl.Context`.
    >>>
    >>> ensm1_fact = max(f.query(ensm1, cl_ctx),
    ...                  key=lambda q: q.giga_op_rate(np.float64))
    >>> ensm2_fact = max(f.query(ensm2, cl_ctx),
    ...                  key=lambda q: q.giga_op_rate(np.float64))


The transformations are called as follows:

.. doctest::

    >>> t_unit = ensm1_fact.transform(t_unit, insn_match="writes:grad_out")
    >>> t_unit = ensm2_fact.transform(t_unit, insn_match="writes:lift_*")

We now ask :mod:`loopy` to generate OpenCL code for the transformed kernel:

.. doctest::

    >>> print(lp.generate_code_v2(t_unit).device_code())   # doctest: +ELLIPSIS
    #define lid(N) ((int) get_local_id(N))
    #define gid(N) ((int) get_group_id(N))
    #if __OPENCL_C_VERSION__ < 120
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
    #endif
    <BLANKLINE>
    __kernel void __attribute__ ((reqd_work_group_size(12, 21, 1))) loopy_kernel(__global double const *__restrict__ D, __global double const *__restrict__ J, int const Nel, __global double *__restrict__ grad_out, __global double const *__restrict__ u)
    {
      __local double D_s_fetch[3 * 12 * 35];
      double acc_jdof_0_tile_jdof_0_inner[3];
      double acc_r;
      double subst_0[3];
      __local double u_s_prftch[21 * 35];
    <BLANKLINE>
      if (-1 + -21 * gid(0) + -1 * lid(1) + Nel >= 0)
        for (int jprftch_u_outer = 0; jprftch_u_outer <= 2 + -1 * lid(0) + (10 + 11 * lid(0)) / 12; ++jprftch_u_outer)
          u_s_prftch[35 * lid(1) + 12 * jprftch_u_outer + lid(0)] = u[35 * (21 * gid(0) + lid(1)) + 12 * jprftch_u_outer + lid(0)];
      for (int idof_0_tile = 0; idof_0_tile <= 2; ++idof_0_tile)
      {
        if (-1 + -1 * lid(1) + -21 * gid(0) + Nel >= 0)
          for (int r_prcmpt_0 = 0; r_prcmpt_0 <= 2; ++r_prcmpt_0)
            if (34 + -1 * lid(0) + -12 * idof_0_tile >= 0)
              acc_jdof_0_tile_jdof_0_inner[r_prcmpt_0] = 0.0;
        barrier(CLK_LOCAL_MEM_FENCE) /* ... */;
        {
          int const iprftchD_outer = 0;
    <BLANKLINE>
          if (34 + -12 * idof_0_tile + -1 * lid(1) >= 0 && 11 + -1 * lid(1) >= 0)
            for (int jprftchD_outer = 0; jprftchD_outer <= 2 + -1 * lid(0) + (10 + 11 * lid(0)) / 12; ++jprftchD_outer)
              for (int rprftchD = 0; rprftchD <= 2; ++rprftchD)
                D_s_fetch[420 * rprftchD + 35 * lid(1) + 12 * jprftchD_outer + lid(0)] = D[1225 * rprftchD + 35 * (12 * idof_0_tile + lid(1)) + 12 * jprftchD_outer + lid(0)];
        }
        barrier(CLK_LOCAL_MEM_FENCE) /* ... */;
        if (-1 + -1 * lid(1) + -21 * gid(0) + Nel >= 0)
        {
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 0] * u_s_prftch[35 * lid(1) + 0];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 0] * u_s_prftch[35 * lid(1) + 0];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 0] * u_s_prftch[35 * lid(1) + 0];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 1] * u_s_prftch[35 * lid(1) + 1];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 1] * u_s_prftch[35 * lid(1) + 1];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 1] * u_s_prftch[35 * lid(1) + 1];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 2] * u_s_prftch[35 * lid(1) + 2];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 2] * u_s_prftch[35 * lid(1) + 2];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 2] * u_s_prftch[35 * lid(1) + 2];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 3] * u_s_prftch[35 * lid(1) + 3];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 3] * u_s_prftch[35 * lid(1) + 3];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 3] * u_s_prftch[35 * lid(1) + 3];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 4] * u_s_prftch[35 * lid(1) + 4];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 4] * u_s_prftch[35 * lid(1) + 4];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 4] * u_s_prftch[35 * lid(1) + 4];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 5] * u_s_prftch[35 * lid(1) + 5];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 5] * u_s_prftch[35 * lid(1) + 5];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 5] * u_s_prftch[35 * lid(1) + 5];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 6] * u_s_prftch[35 * lid(1) + 6];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 6] * u_s_prftch[35 * lid(1) + 6];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 6] * u_s_prftch[35 * lid(1) + 6];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 7] * u_s_prftch[35 * lid(1) + 7];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 7] * u_s_prftch[35 * lid(1) + 7];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 7] * u_s_prftch[35 * lid(1) + 7];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 8] * u_s_prftch[35 * lid(1) + 8];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 8] * u_s_prftch[35 * lid(1) + 8];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 8] * u_s_prftch[35 * lid(1) + 8];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 9] * u_s_prftch[35 * lid(1) + 9];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 9] * u_s_prftch[35 * lid(1) + 9];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 9] * u_s_prftch[35 * lid(1) + 9];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 10] * u_s_prftch[35 * lid(1) + 10];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 10] * u_s_prftch[35 * lid(1) + 10];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 10] * u_s_prftch[35 * lid(1) + 10];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 11] * u_s_prftch[35 * lid(1) + 11];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 11] * u_s_prftch[35 * lid(1) + 11];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 11] * u_s_prftch[35 * lid(1) + 11];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 12] * u_s_prftch[35 * lid(1) + 12];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 12] * u_s_prftch[35 * lid(1) + 12];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 12] * u_s_prftch[35 * lid(1) + 12];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 13] * u_s_prftch[35 * lid(1) + 13];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 13] * u_s_prftch[35 * lid(1) + 13];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 13] * u_s_prftch[35 * lid(1) + 13];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 14] * u_s_prftch[35 * lid(1) + 14];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 14] * u_s_prftch[35 * lid(1) + 14];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 14] * u_s_prftch[35 * lid(1) + 14];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 15] * u_s_prftch[35 * lid(1) + 15];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 15] * u_s_prftch[35 * lid(1) + 15];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 15] * u_s_prftch[35 * lid(1) + 15];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 16] * u_s_prftch[35 * lid(1) + 16];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 16] * u_s_prftch[35 * lid(1) + 16];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 16] * u_s_prftch[35 * lid(1) + 16];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 17] * u_s_prftch[35 * lid(1) + 17];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 17] * u_s_prftch[35 * lid(1) + 17];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 17] * u_s_prftch[35 * lid(1) + 17];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 18] * u_s_prftch[35 * lid(1) + 18];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 18] * u_s_prftch[35 * lid(1) + 18];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 18] * u_s_prftch[35 * lid(1) + 18];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 19] * u_s_prftch[35 * lid(1) + 19];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 19] * u_s_prftch[35 * lid(1) + 19];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 19] * u_s_prftch[35 * lid(1) + 19];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 20] * u_s_prftch[35 * lid(1) + 20];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 20] * u_s_prftch[35 * lid(1) + 20];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 20] * u_s_prftch[35 * lid(1) + 20];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 21] * u_s_prftch[35 * lid(1) + 21];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 21] * u_s_prftch[35 * lid(1) + 21];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 21] * u_s_prftch[35 * lid(1) + 21];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 22] * u_s_prftch[35 * lid(1) + 22];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 22] * u_s_prftch[35 * lid(1) + 22];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 22] * u_s_prftch[35 * lid(1) + 22];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 23] * u_s_prftch[35 * lid(1) + 23];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 23] * u_s_prftch[35 * lid(1) + 23];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 23] * u_s_prftch[35 * lid(1) + 23];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 24] * u_s_prftch[35 * lid(1) + 24];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 24] * u_s_prftch[35 * lid(1) + 24];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 24] * u_s_prftch[35 * lid(1) + 24];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 25] * u_s_prftch[35 * lid(1) + 25];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 25] * u_s_prftch[35 * lid(1) + 25];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 25] * u_s_prftch[35 * lid(1) + 25];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 26] * u_s_prftch[35 * lid(1) + 26];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 26] * u_s_prftch[35 * lid(1) + 26];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 26] * u_s_prftch[35 * lid(1) + 26];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 27] * u_s_prftch[35 * lid(1) + 27];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 27] * u_s_prftch[35 * lid(1) + 27];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 27] * u_s_prftch[35 * lid(1) + 27];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 28] * u_s_prftch[35 * lid(1) + 28];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 28] * u_s_prftch[35 * lid(1) + 28];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 28] * u_s_prftch[35 * lid(1) + 28];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 29] * u_s_prftch[35 * lid(1) + 29];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 29] * u_s_prftch[35 * lid(1) + 29];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 29] * u_s_prftch[35 * lid(1) + 29];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 30] * u_s_prftch[35 * lid(1) + 30];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 30] * u_s_prftch[35 * lid(1) + 30];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 30] * u_s_prftch[35 * lid(1) + 30];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 31] * u_s_prftch[35 * lid(1) + 31];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 31] * u_s_prftch[35 * lid(1) + 31];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 31] * u_s_prftch[35 * lid(1) + 31];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 32] * u_s_prftch[35 * lid(1) + 32];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 32] * u_s_prftch[35 * lid(1) + 32];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 32] * u_s_prftch[35 * lid(1) + 32];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 33] * u_s_prftch[35 * lid(1) + 33];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 33] * u_s_prftch[35 * lid(1) + 33];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 33] * u_s_prftch[35 * lid(1) + 33];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[0] = acc_jdof_0_tile_jdof_0_inner[0] + D_s_fetch[420 * 0 + 35 * lid(0) + 34] * u_s_prftch[35 * lid(1) + 34];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[1] = acc_jdof_0_tile_jdof_0_inner[1] + D_s_fetch[420 * 1 + 35 * lid(0) + 34] * u_s_prftch[35 * lid(1) + 34];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            acc_jdof_0_tile_jdof_0_inner[2] = acc_jdof_0_tile_jdof_0_inner[2] + D_s_fetch[420 * 2 + 35 * lid(0) + 34] * u_s_prftch[35 * lid(1) + 34];
        }
        if (-1 + -1 * lid(1) + -21 * gid(0) + Nel >= 0)
        {
          for (int r_prcmpt_1 = 0; r_prcmpt_1 <= 2; ++r_prcmpt_1)
            if (34 + -1 * lid(0) + -12 * idof_0_tile >= 0)
              subst_0[r_prcmpt_1] = acc_jdof_0_tile_jdof_0_inner[r_prcmpt_1];
          if (34 + -12 * idof_0_tile + -1 * lid(0) >= 0)
            for (int x = 0; x <= 2; ++x)
            {
              acc_r = 0.0;
              acc_r = acc_r + subst_0[0] * J[Nel * 3 * x + Nel * 0 + 21 * gid(0) + lid(1)];
              acc_r = acc_r + subst_0[1] * J[Nel * 3 * x + Nel * 1 + 21 * gid(0) + lid(1)];
              acc_r = acc_r + subst_0[2] * J[Nel * 3 * x + Nel * 2 + 21 * gid(0) + lid(1)];
              grad_out[35 * Nel * x + 35 * (21 * gid(0) + lid(1)) + 12 * idof_0_tile + lid(0)] = acc_r;
            }
        }
      }
    }
    <BLANKLINE>
    __kernel void __attribute__ ((reqd_work_group_size(8, 8, 1))) loopy_kernel_0(__global double const *__restrict__ F_0, __global double const *__restrict__ F_1, __global double const *__restrict__ F_2, __global double const *__restrict__ F_3, __global double const *__restrict__ Jface, __global double const *__restrict__ L, int const Nel, __global double *__restrict__ lift_0, __global double *__restrict__ lift_1, __global double *__restrict__ lift_2, __global double *__restrict__ lift_3)
    {
      __local double L_s_fetch[35 * 4 * 15];
      double acc_iface_0_ifacedof_0;
      double acc_iface_0_ifacedof_1;
      double acc_iface_0_ifacedof_2;
      double acc_iface_0_ifacedof_3;
      __local double prcmpt_stage1[4 * 8 * 15];
      __local double prcmpt_stage1_0[4 * 8 * 15];
      __local double prcmpt_stage1_1[4 * 8 * 15];
      __local double prcmpt_stage1_2[4 * 8 * 15];
    <BLANKLINE>
      for (int ifaceprftchL_s = 0; ifaceprftchL_s <= 3; ++ifaceprftchL_s)
        L_s_fetch[60 * (8 * 0 + lid(1)) + 15 * ifaceprftchL_s + 8 * 0 + lid(0)] = L[60 * (8 * 0 + lid(1)) + 15 * ifaceprftchL_s + 8 * 0 + lid(0)];
      if (6 + -1 * lid(0) >= 0)
        for (int ifaceprftchL_s = 0; ifaceprftchL_s <= 3; ++ifaceprftchL_s)
          L_s_fetch[60 * (8 * 0 + lid(1)) + 15 * ifaceprftchL_s + 8 * 1 + lid(0)] = L[60 * (8 * 0 + lid(1)) + 15 * ifaceprftchL_s + 8 * 1 + lid(0)];
      for (int ifaceprftchL_s = 0; ifaceprftchL_s <= 3; ++ifaceprftchL_s)
        L_s_fetch[60 * (8 * 1 + lid(1)) + 15 * ifaceprftchL_s + 8 * 0 + lid(0)] = L[60 * (8 * 1 + lid(1)) + 15 * ifaceprftchL_s + 8 * 0 + lid(0)];
      if (6 + -1 * lid(0) >= 0)
        for (int ifaceprftchL_s = 0; ifaceprftchL_s <= 3; ++ifaceprftchL_s)
          L_s_fetch[60 * (8 * 1 + lid(1)) + 15 * ifaceprftchL_s + 8 * 1 + lid(0)] = L[60 * (8 * 1 + lid(1)) + 15 * ifaceprftchL_s + 8 * 1 + lid(0)];
      for (int ifaceprftchL_s = 0; ifaceprftchL_s <= 3; ++ifaceprftchL_s)
        L_s_fetch[60 * (8 * 2 + lid(1)) + 15 * ifaceprftchL_s + 8 * 0 + lid(0)] = L[60 * (8 * 2 + lid(1)) + 15 * ifaceprftchL_s + 8 * 0 + lid(0)];
      if (6 + -1 * lid(0) >= 0)
        for (int ifaceprftchL_s = 0; ifaceprftchL_s <= 3; ++ifaceprftchL_s)
          L_s_fetch[60 * (8 * 2 + lid(1)) + 15 * ifaceprftchL_s + 8 * 1 + lid(0)] = L[60 * (8 * 2 + lid(1)) + 15 * ifaceprftchL_s + 8 * 1 + lid(0)];
      for (int ifaceprftchL_s = 0; ifaceprftchL_s <= 3; ++ifaceprftchL_s)
        L_s_fetch[60 * (8 * 3 + lid(1)) + 15 * ifaceprftchL_s + 8 * 0 + lid(0)] = L[60 * (8 * 3 + lid(1)) + 15 * ifaceprftchL_s + 8 * 0 + lid(0)];
      if (6 + -1 * lid(0) >= 0)
        for (int ifaceprftchL_s = 0; ifaceprftchL_s <= 3; ++ifaceprftchL_s)
          L_s_fetch[60 * (8 * 3 + lid(1)) + 15 * ifaceprftchL_s + 8 * 1 + lid(0)] = L[60 * (8 * 3 + lid(1)) + 15 * ifaceprftchL_s + 8 * 1 + lid(0)];
      if (2 + -1 * lid(1) >= 0)
      {
        for (int ifaceprftchL_s = 0; ifaceprftchL_s <= 3; ++ifaceprftchL_s)
          L_s_fetch[60 * (8 * 4 + lid(1)) + 15 * ifaceprftchL_s + 8 * 0 + lid(0)] = L[60 * (8 * 4 + lid(1)) + 15 * ifaceprftchL_s + 8 * 0 + lid(0)];
        if (6 + -1 * lid(0) >= 0)
          for (int ifaceprftchL_s = 0; ifaceprftchL_s <= 3; ++ifaceprftchL_s)
            L_s_fetch[60 * (8 * 4 + lid(1)) + 15 * ifaceprftchL_s + 8 * 1 + lid(0)] = L[60 * (8 * 4 + lid(1)) + 15 * ifaceprftchL_s + 8 * 1 + lid(0)];
      }
      if (-1 + -8 * gid(0) + -1 * lid(1) + Nel >= 0)
        for (int iface_prcmpt_stage1 = 0; iface_prcmpt_stage1 <= 3; ++iface_prcmpt_stage1)
          for (int ifacedof_prcmpt_stage1_outer = 0; ifacedof_prcmpt_stage1_outer <= 1 + -1 * lid(0) + (6 + 7 * lid(0)) / 8; ++ifacedof_prcmpt_stage1_outer)
          {
            prcmpt_stage1[120 * iface_prcmpt_stage1 + 15 * lid(1) + 8 * ifacedof_prcmpt_stage1_outer + lid(0)] = Jface[Nel * iface_prcmpt_stage1 + 8 * gid(0) + lid(1)] * F_0[15 * Nel * iface_prcmpt_stage1 + 15 * (8 * gid(0) + lid(1)) + 8 * ifacedof_prcmpt_stage1_outer + lid(0)];
            prcmpt_stage1_0[120 * iface_prcmpt_stage1 + 15 * lid(1) + 8 * ifacedof_prcmpt_stage1_outer + lid(0)] = Jface[Nel * iface_prcmpt_stage1 + 8 * gid(0) + lid(1)] * F_1[15 * Nel * iface_prcmpt_stage1 + 15 * (8 * gid(0) + lid(1)) + 8 * ifacedof_prcmpt_stage1_outer + lid(0)];
            prcmpt_stage1_1[120 * iface_prcmpt_stage1 + 15 * lid(1) + 8 * ifacedof_prcmpt_stage1_outer + lid(0)] = Jface[Nel * iface_prcmpt_stage1 + 8 * gid(0) + lid(1)] * F_2[15 * Nel * iface_prcmpt_stage1 + 15 * (8 * gid(0) + lid(1)) + 8 * ifacedof_prcmpt_stage1_outer + lid(0)];
            prcmpt_stage1_2[120 * iface_prcmpt_stage1 + 15 * lid(1) + 8 * ifacedof_prcmpt_stage1_outer + lid(0)] = Jface[Nel * iface_prcmpt_stage1 + 8 * gid(0) + lid(1)] * F_3[15 * Nel * iface_prcmpt_stage1 + 15 * (8 * gid(0) + lid(1)) + 8 * ifacedof_prcmpt_stage1_outer + lid(0)];
          }
      barrier(CLK_LOCAL_MEM_FENCE) /* ... */;
      if (-1 + -8 * gid(0) + -1 * lid(1) + Nel >= 0)
      {
        acc_iface_0_ifacedof_0 = 0.0;
        acc_iface_0_ifacedof_1 = 0.0;
        acc_iface_0_ifacedof_2 = 0.0;
        acc_iface_0_ifacedof_3 = 0.0;
        for (int ifacedof_0 = 0; ifacedof_0 <= 14; ++ifacedof_0)
          for (int iface_0 = 0; iface_0 <= 3; ++iface_0)
          {
            acc_iface_0_ifacedof_0 = acc_iface_0_ifacedof_0 + prcmpt_stage1[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 0 + lid(0)) + 15 * iface_0 + ifacedof_0];
            acc_iface_0_ifacedof_1 = acc_iface_0_ifacedof_1 + prcmpt_stage1_0[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 0 + lid(0)) + 15 * iface_0 + ifacedof_0];
            acc_iface_0_ifacedof_2 = acc_iface_0_ifacedof_2 + prcmpt_stage1_1[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 0 + lid(0)) + 15 * iface_0 + ifacedof_0];
            acc_iface_0_ifacedof_3 = acc_iface_0_ifacedof_3 + prcmpt_stage1_2[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 0 + lid(0)) + 15 * iface_0 + ifacedof_0];
          }
        lift_0[35 * (8 * gid(0) + lid(1)) + 8 * 0 + lid(0)] = acc_iface_0_ifacedof_0;
        lift_1[35 * (8 * gid(0) + lid(1)) + 8 * 0 + lid(0)] = acc_iface_0_ifacedof_1;
        lift_2[35 * (8 * gid(0) + lid(1)) + 8 * 0 + lid(0)] = acc_iface_0_ifacedof_2;
        lift_3[35 * (8 * gid(0) + lid(1)) + 8 * 0 + lid(0)] = acc_iface_0_ifacedof_3;
        acc_iface_0_ifacedof_0 = 0.0;
        acc_iface_0_ifacedof_1 = 0.0;
        acc_iface_0_ifacedof_2 = 0.0;
        acc_iface_0_ifacedof_3 = 0.0;
        for (int ifacedof_0 = 0; ifacedof_0 <= 14; ++ifacedof_0)
          for (int iface_0 = 0; iface_0 <= 3; ++iface_0)
          {
            acc_iface_0_ifacedof_0 = acc_iface_0_ifacedof_0 + prcmpt_stage1[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 1 + lid(0)) + 15 * iface_0 + ifacedof_0];
            acc_iface_0_ifacedof_1 = acc_iface_0_ifacedof_1 + prcmpt_stage1_0[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 1 + lid(0)) + 15 * iface_0 + ifacedof_0];
            acc_iface_0_ifacedof_2 = acc_iface_0_ifacedof_2 + prcmpt_stage1_1[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 1 + lid(0)) + 15 * iface_0 + ifacedof_0];
            acc_iface_0_ifacedof_3 = acc_iface_0_ifacedof_3 + prcmpt_stage1_2[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 1 + lid(0)) + 15 * iface_0 + ifacedof_0];
          }
        lift_0[35 * (8 * gid(0) + lid(1)) + 8 * 1 + lid(0)] = acc_iface_0_ifacedof_0;
        lift_1[35 * (8 * gid(0) + lid(1)) + 8 * 1 + lid(0)] = acc_iface_0_ifacedof_1;
        lift_2[35 * (8 * gid(0) + lid(1)) + 8 * 1 + lid(0)] = acc_iface_0_ifacedof_2;
        lift_3[35 * (8 * gid(0) + lid(1)) + 8 * 1 + lid(0)] = acc_iface_0_ifacedof_3;
        acc_iface_0_ifacedof_0 = 0.0;
        acc_iface_0_ifacedof_1 = 0.0;
        acc_iface_0_ifacedof_2 = 0.0;
        acc_iface_0_ifacedof_3 = 0.0;
        for (int ifacedof_0 = 0; ifacedof_0 <= 14; ++ifacedof_0)
          for (int iface_0 = 0; iface_0 <= 3; ++iface_0)
          {
            acc_iface_0_ifacedof_0 = acc_iface_0_ifacedof_0 + prcmpt_stage1[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 2 + lid(0)) + 15 * iface_0 + ifacedof_0];
            acc_iface_0_ifacedof_1 = acc_iface_0_ifacedof_1 + prcmpt_stage1_0[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 2 + lid(0)) + 15 * iface_0 + ifacedof_0];
            acc_iface_0_ifacedof_2 = acc_iface_0_ifacedof_2 + prcmpt_stage1_1[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 2 + lid(0)) + 15 * iface_0 + ifacedof_0];
            acc_iface_0_ifacedof_3 = acc_iface_0_ifacedof_3 + prcmpt_stage1_2[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 2 + lid(0)) + 15 * iface_0 + ifacedof_0];
          }
        lift_0[35 * (8 * gid(0) + lid(1)) + 8 * 2 + lid(0)] = acc_iface_0_ifacedof_0;
        lift_1[35 * (8 * gid(0) + lid(1)) + 8 * 2 + lid(0)] = acc_iface_0_ifacedof_1;
        lift_2[35 * (8 * gid(0) + lid(1)) + 8 * 2 + lid(0)] = acc_iface_0_ifacedof_2;
        lift_3[35 * (8 * gid(0) + lid(1)) + 8 * 2 + lid(0)] = acc_iface_0_ifacedof_3;
        acc_iface_0_ifacedof_0 = 0.0;
        acc_iface_0_ifacedof_1 = 0.0;
        acc_iface_0_ifacedof_2 = 0.0;
        acc_iface_0_ifacedof_3 = 0.0;
        for (int ifacedof_0 = 0; ifacedof_0 <= 14; ++ifacedof_0)
          for (int iface_0 = 0; iface_0 <= 3; ++iface_0)
          {
            acc_iface_0_ifacedof_0 = acc_iface_0_ifacedof_0 + prcmpt_stage1[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 3 + lid(0)) + 15 * iface_0 + ifacedof_0];
            acc_iface_0_ifacedof_1 = acc_iface_0_ifacedof_1 + prcmpt_stage1_0[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 3 + lid(0)) + 15 * iface_0 + ifacedof_0];
            acc_iface_0_ifacedof_2 = acc_iface_0_ifacedof_2 + prcmpt_stage1_1[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 3 + lid(0)) + 15 * iface_0 + ifacedof_0];
            acc_iface_0_ifacedof_3 = acc_iface_0_ifacedof_3 + prcmpt_stage1_2[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 3 + lid(0)) + 15 * iface_0 + ifacedof_0];
          }
        lift_0[35 * (8 * gid(0) + lid(1)) + 8 * 3 + lid(0)] = acc_iface_0_ifacedof_0;
        lift_1[35 * (8 * gid(0) + lid(1)) + 8 * 3 + lid(0)] = acc_iface_0_ifacedof_1;
        lift_2[35 * (8 * gid(0) + lid(1)) + 8 * 3 + lid(0)] = acc_iface_0_ifacedof_2;
        lift_3[35 * (8 * gid(0) + lid(1)) + 8 * 3 + lid(0)] = acc_iface_0_ifacedof_3;
        if (2 + -1 * lid(0) >= 0)
        {
          acc_iface_0_ifacedof_0 = 0.0;
          acc_iface_0_ifacedof_1 = 0.0;
          acc_iface_0_ifacedof_2 = 0.0;
          acc_iface_0_ifacedof_3 = 0.0;
          for (int ifacedof_0 = 0; ifacedof_0 <= 14; ++ifacedof_0)
            for (int iface_0 = 0; iface_0 <= 3; ++iface_0)
            {
              acc_iface_0_ifacedof_0 = acc_iface_0_ifacedof_0 + prcmpt_stage1[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 4 + lid(0)) + 15 * iface_0 + ifacedof_0];
              acc_iface_0_ifacedof_1 = acc_iface_0_ifacedof_1 + prcmpt_stage1_0[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 4 + lid(0)) + 15 * iface_0 + ifacedof_0];
              acc_iface_0_ifacedof_2 = acc_iface_0_ifacedof_2 + prcmpt_stage1_1[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 4 + lid(0)) + 15 * iface_0 + ifacedof_0];
              acc_iface_0_ifacedof_3 = acc_iface_0_ifacedof_3 + prcmpt_stage1_2[120 * iface_0 + 15 * lid(1) + ifacedof_0] * L_s_fetch[60 * (8 * 4 + lid(0)) + 15 * iface_0 + ifacedof_0];
            }
          lift_0[35 * (8 * gid(0) + lid(1)) + 8 * 4 + lid(0)] = acc_iface_0_ifacedof_0;
          lift_1[35 * (8 * gid(0) + lid(1)) + 8 * 4 + lid(0)] = acc_iface_0_ifacedof_1;
          lift_2[35 * (8 * gid(0) + lid(1)) + 8 * 4 + lid(0)] = acc_iface_0_ifacedof_2;
          lift_3[35 * (8 * gid(0) + lid(1)) + 8 * 4 + lid(0)] = acc_iface_0_ifacedof_3;
        }
      }
    }
