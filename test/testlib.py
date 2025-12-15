import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

import loopy as lp
import numpy as np
from numpy.random import Generator

import feinsum as fnsm
import feinsum.loopy_utils as lp_utils

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def transform_3d_p4_grad(t_unit, insn_match=None, kernel_name=None):
    """
    A transformation that performs ~1.8 TFlOps/s on a TitanV.
    """
    from loopy.match import parse_match
    from pymbolic import variables

    insn_match = parse_match(insn_match)

    # {{{ define ref_einsum; get subst_map

    ndim = 3
    ndofs = 35

    ref_einsum = fnsm.einsum(
        "xer,rij,ej->xei",
        fnsm.array(
            "J",
            (
                ndim,
                "E",
                ndim,
            ),
        ),
        fnsm.array("R", (ndim, ndofs, ndofs)),
        fnsm.array("u", ("E", ndofs)),
    )

    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit, ref_einsum, insn_match=insn_match
    )

    # }}}

    # {{{ transform parameters

    ncells_per_workgroup = 9
    nworkitems_per_cell = 7
    j_tile_len = 9
    i_tile_len = 35

    # }}}

    # {{{ read variables

    i = subst_map["i"]
    j = subst_map["j"]
    e = subst_map["e"]
    R = subst_map["R"]
    J = subst_map["J"]
    u = subst_map["u"]
    x = subst_map["x"]
    r = subst_map["r"]
    out = subst_map["_fe_out"]
    e_inner, e_outer = f"{e}_inner", f"{e}_outer"
    i_inner, i_outer = f"{i}_inner", f"{i}_outer"
    j_inner = f"{j}_inner"
    i_tile, j_tile = f"{i}_tile", f"{j}_tile"
    J_prftch = f"{J}_prftch"
    e_prftch = f"{e}_prftch"
    j_prftch = f"{j}_prftch"
    i_prftch = f"{i}_prftch"
    r_prftch = f"{r}_prftch"
    r_prcmpt = f"{r}_prcmpt"

    # }}}

    # {{{ term hoisting to match the flop count of opt_einsum

    from loopy.symbolic import Reduction

    knl = t_unit.default_entrypoint

    knl = lp.split_reduction_inward(knl, j)
    knl = lp_utils.hoist_invariant_multiplicative_terms_in_sum_reduction(knl, j)
    knl = lp_utils.extract_multiplicative_terms_in_sum_reduction_as_subst(
        knl,
        within=None,
        subst_name="subst",
        arguments=variables(f"{r} {i} {e}"),
        terms_filter=lambda x: isinstance(x, Reduction),
    )

    t_unit = t_unit.with_kernel(knl)

    # }}}

    t_unit = lp.split_iname(t_unit, i, i_tile_len, outer_iname=i_tile)
    t_unit = lp.split_iname(t_unit, j, j_tile_len, outer_iname=j_tile)

    t_unit = lp.rename_iname(t_unit, i_inner, i)
    t_unit = lp.rename_iname(t_unit, j_inner, j)
    t_unit = lp.split_iname(
        t_unit, e, ncells_per_workgroup, outer_tag="g.0", inner_tag="l.1"
    )
    t_unit = lp.split_iname(t_unit, i, nworkitems_per_cell, inner_tag="l.0")

    t_unit = lp.precompute(
        t_unit,
        J,
        sweep_inames=[r, x],
        precompute_outer_inames=frozenset([e_inner, e_outer, i_inner]),
        temporary_address_space=lp.AddressSpace.PRIVATE,
        temporary_name=J_prftch,
    )

    # {{{ TODO: Make precompute smarter (should be a single precompute call)

    t_unit = lp.precompute(
        t_unit,
        "subst",
        sweep_inames=[r],
        precompute_outer_inames=frozenset(
            {e_inner, e_outer, i_inner, i_outer, i_tile}
        ),
        precompute_inames=r_prcmpt,
        temporary_name="tmp_hoist",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        compute_insn_id="insn_hoist",
    )
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit, i_outer, only_var_names=["tmp_hoist"]
    )

    t_unit = lp.duplicate_inames(t_unit, i_outer, "id:insn_hoist", "i_outer_hoist")

    # }}}

    # {{{ Move 'u ' to shared.

    # Prefetch 'u' within the tile
    t_unit = lp.precompute(
        t_unit,
        u,
        sweep_inames=[e_inner, j],
        precompute_outer_inames=frozenset([e_outer, i_tile, j_tile]),
        temporary_address_space=lp.AddressSpace.LOCAL,
        precompute_inames=[e_prftch, j_prftch],
        default_tag=None,
    )

    t_unit = lp.split_iname(
        t_unit, e_prftch, ncells_per_workgroup, inner_tag="l.1", outer_tag="unr"
    )
    t_unit = lp.split_iname(
        t_unit, j_prftch, nworkitems_per_cell, inner_tag="l.0", outer_tag="unr"
    )

    # }}}

    # {{{ Move 'R' to shared.

    t_unit = lp.precompute(
        t_unit,
        R,
        sweep_inames=[r_prcmpt, i_inner, "i_outer_hoist", j],
        precompute_outer_inames=frozenset([e_outer, i_tile, j_tile]),
        temporary_address_space=lp.AddressSpace.LOCAL,
        precompute_inames=[r_prftch, i_prftch, j_prftch],
        default_tag=None,
        within="id:insn_hoist",
    )

    if 0:
        # This branch improves perf. by 20% but, something in loopy
        # non-deterministically leads to very ugly domains.
        t_unit = lp.join_inames(t_unit, [r_prftch, i_prftch, j_prftch], "i_Rprftch")

        t_unit = lp.split_iname(
            t_unit,
            "i_Rprftch",
            ncells_per_workgroup * nworkitems_per_cell,
            outer_tag="unr",
        )

        t_unit = lp.split_iname(
            t_unit,
            "i_Rprftch_inner",
            nworkitems_per_cell,
            inner_tag="l.0",
            outer_tag="l.1",
        )
    else:
        t_unit = lp.split_iname(
            t_unit, i_prftch, ncells_per_workgroup, inner_tag="l.1", outer_tag="unr"
        )
        t_unit = lp.split_iname(
            t_unit, j_prftch, nworkitems_per_cell, inner_tag="l.0", outer_tag="unr"
        )

    # }}}

    # {{{ make buffer array smarter (should be a single call to buffer_array)

    t_unit = lp.buffer_array(
        t_unit,
        out,
        buffer_inames=[x],
        init_expression="0",
        default_tag=None,
        temporary_scope=lp.AddressSpace.PRIVATE,
    )
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit, i_outer, only_var_names={f"{out}_buf"}
    )

    t_unit = lp.duplicate_inames(
        t_unit,
        inames=[i_outer],
        within=f"id:init_{out}",
        new_inames=[f"{out}_init_1"],
    )

    t_unit = lp.duplicate_inames(
        t_unit,
        inames=[i_outer],
        within=f"id:store_{out}",
        new_inames=[f"{out}_store_1"],
    )

    # }}}

    # {{{ must be smarter way of doing this in loopy

    t_unit = lp.realize_reduction(t_unit, insn_id_filter="insn_hoist")
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit,
        frozenset([r_prcmpt, "i_outer_hoist"]),
        only_var_names={f"acc_{j_tile}_{j}"},
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        ["i_outer_hoist", r_prcmpt],
        within=f"id:insn_hoist_{j_tile}_{j}_init",
        new_inames=["i_outer_hoist_init", f"{r_prcmpt}_init"],
    )

    t_unit = lp.duplicate_inames(
        t_unit,
        ["i_outer_hoist", r_prcmpt],
        within="id:insn_hoist",
        new_inames=["i_outer_hoist_store", f"{r_prcmpt}_store"],
    )

    # }}}

    return t_unit


# {{{ generate random batched einsum.


def _operand_names_iter():
    for i in range(26):
        yield chr(ord("A") + i)

    for i in range(26):
        for j in range(26):
            yield (chr(ord("A") + i) + chr(ord("A") + j))


def generate_batched_einsum(
    rng: Generator,
    *,
    max_dim_size=7,
) -> fnsm.BatchedEinsum:

    b = rng.integers(low=1, high=17).item()
    n = rng.integers(low=1, high=9).item()
    n_free_indices = rng.integers(low=1, high=8).item()
    n_redn_indices = rng.integers(low=0, high=8).item()

    assert (
        n_free_indices + n_redn_indices
    ) < 27, "Can support only 26 indices in einsum."
    output_idx_list = tuple(chr(97 + (i + 8) % 26) for i in range(n_free_indices))
    redn_indices = tuple(
        chr(97 + (i + n_free_indices + 8) % 26) for i in range(n_redn_indices)
    )
    operand_names_iter = _operand_names_iter()

    def operand_names_gen(): return next(operand_names_iter)

    all_indices = output_idx_list + redn_indices
    all_lengths = [4, 8, 16, 32, 64]
    all_dtypes = [np.float16, np.float32, np.float64]

    input_idx_lists = tuple(
        tuple(
            rng.choice(all_indices).item()
            for _ in range(rng.integers(low=0, high=max_dim_size + 1))
        )
        for _ in range(n)
    )

    while not (
        set(sum(input_idx_lists, start=cast("tuple[str]", ())))
        >= set(output_idx_list)
    ):
        # Input indices does not contain all the output indices => this is
        # invalid for einsum grammar. Regenerate it.
        input_idx_lists = tuple(
            tuple(
                rng.choice(all_indices).item()
                for _ in range(rng.integers(low=0, high=max_dim_size + 1))
            )
            for _ in range(n)
        )

    index_to_length = {idx: rng.choice(all_lengths).item() for idx in all_indices}
    operand_pos_to_dtype = tuple(
        tuple(np.dtype(rng.choice(all_dtypes)) for _ in range(n))
        for _ in range(b)
    )
    shape_x_dtype_to_names: dict[
        tuple[tuple[int, ...], np.dtype[Any]], list[str]
    ] = {}

    arg_names: list[list[str]] = []
    arg_to_dtype: dict[str, np.dtype[Any]] = {}
    arg_to_shape: dict[str, tuple[int, ...]] = {}
    for i in range(b):
        arg_row: list[str] = []
        for j, idx_list in enumerate(input_idx_lists):
            shape = tuple(index_to_length[idx] for idx in idx_list)
            dtype = operand_pos_to_dtype[i][j]
            try:
                potential_repeats = shape_x_dtype_to_names[shape, dtype]
            except KeyError:
                new_name = operand_names_gen()
                arg_row.append(new_name)
                shape_x_dtype_to_names[shape, dtype] = [new_name]
                arg_to_dtype[new_name] = dtype
                arg_to_shape[new_name] = shape
            else:
                should_repeat = rng.choice([False, True], p=[0.3, 0.7]).item()
                if should_repeat:
                    arg_row.append(rng.choice(potential_repeats).item())
                else:
                    new_name = operand_names_gen()
                    arg_row.append(new_name)
                    potential_repeats.append(new_name)
                    arg_to_dtype[new_name] = dtype
                    arg_to_shape[new_name] = shape

        arg_names.append(arg_row)
        del arg_row

    return fnsm.batched_einsum(
        f"{','.join([''.join(idx_list) for idx_list in input_idx_lists])}"
        f" -> {''.join(output_idx_list)}",
        [
            [
                fnsm.array(
                    arg_name,
                    arg_to_shape[arg_name],
                    arg_to_dtype[arg_name],
                )
                for arg_name in arg_row
            ]
            for arg_row in arg_names
        ],
    )

# }}}

# {{{ apply substitution mapping to batched einsum.


def apply_renaming_to_batched_einsum(
    e1: fnsm.BatchedEinsum,
    sigma_i: Sequence[int],
    sigma_j: Sequence[int],
    sigma_idx: Mapping[str, str],
    sigma_arg: Mapping[str, str],
) -> fnsm.BatchedEinsum:
    """
    Returns a batched einsum, *e2*, that is isomorphic to *e1*, such that:

    - Argument at ``i, j`` in e2 will be
      ``sigma_arg[e1.arg_names[sigma_i[i], sigma_j[j]]]``
    """
    assert sorted(sigma_i) == list(
        range(len(sigma_i))
    ), "sigma_i must be a valid permutation."
    assert sorted(sigma_j) == list(
        range(len(sigma_j))
    ), "sigma_j must be a valid permutation."

    e2_out_idx_list = tuple(sigma_idx[idx] for idx in e1.out_idx_set)
    e2_in_idx_lists = tuple(
        tuple(sigma_idx[idx] for idx in e1.in_idx_sets[j]) for j in sigma_j
    )

    return fnsm.batched_einsum(
        f"{','.join([''.join(idx_list) for idx_list in e2_in_idx_lists])}"
        f" -> {''.join(e2_out_idx_list)}",
        [
            [e1.args[i][j].copy(name=sigma_arg[e1.args[i][j].name]) for j in sigma_j]
            for i in sigma_i
        ],
    )


# }}}
