import os

import pyopencl as cl

import feinsum as fnsm

NEW_DB_NAME = "transform_archive_v5.db"
TOP_K = 10  # how many entries to reevaluate


def main():
    from feinsum.sql_utils import get_timed_einsums_in_db, record_into_db
    from feinsum.tuning import _get_impls_path

    cl_ctx = cl.create_some_context()
    cq = cl.CommandQueue(cl_ctx)

    for i, einsum in enumerate(get_timed_einsums_in_db(cq.device)):
        best_queries = sorted(
            fnsm.query(einsum, cl_ctx), key=lambda q: -q.giga_op_rate("float64")
        )[:TOP_K]
        if os.path.exists(NEW_DB_NAME):
            recorded_queries = fnsm.query(einsum, cl_ctx, database=NEW_DB_NAME)
            recorded_transform_params = [
                q.transform_params for q in recorded_queries
            ]
        else:
            recorded_transform_params = []

        print(i, len(best_queries))
        for iquery, query in enumerate(best_queries):
            if query.transform_params in recorded_transform_params:
                print("Skipping as completed")
                continue
            print(
                f"({iquery}, {query.transform_id}):"
                f" Old GFLOPS/s = {query.giga_op_rate('float64')}."
            )
            print(query.transform_params)
            record_into_db(
                einsum,
                cl_ctx,
                module_path=os.path.join(_get_impls_path(), query.transform_id),
                transform_params=query.transform_params,
                database=NEW_DB_NAME,
            )
            print(75 * "=")


if __name__ == "__main__":
    main()
