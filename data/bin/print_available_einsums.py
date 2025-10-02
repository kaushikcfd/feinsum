import pyopencl as cl

import feinsum as fnsm


def main():
    from feinsum.sql_utils import get_timed_einsums_in_db

    cl_ctx = cl.create_some_context()
    cq = cl.CommandQueue(cl_ctx)
    einsums_in_db = get_timed_einsums_in_db(cq.device)
    print(f"Total einsums in DB = {len(einsums_in_db)}")

    for einsum in einsums_in_db:
        facts = fnsm.query(einsum, cl_ctx)
        print(f"Einsum: {einsum}, Available data: {len(facts)}")


if __name__ == "__main__":
    main()
