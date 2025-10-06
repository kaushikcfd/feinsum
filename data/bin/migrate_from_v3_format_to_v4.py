import json
import sqlite3 as sql

import numpy as np

import feinsum as f


def record_into_db(
    cursor: sql.Cursor,
    *,
    einsum: f.BatchedEinsum,
    device_name: str,
    transform_id: str,
    transform_params: str,
    runtime_in_sec: float,
    compiler_version: str,
    giga_op_info: str,
    timestamp: str,
):
    from feinsum.sql_utils import (
        dump_index_to_length,
        dump_use_matrix,
        dump_value_to_dtype,
    )

    einsum = f.canonicalize_einsum(einsum)
    subscripts = einsum.get_subscripts()
    index_to_length = dump_index_to_length(einsum)
    use_matrix = dump_use_matrix(einsum)
    value_to_dtype = dump_value_to_dtype(einsum)
    cursor.execute(
        "INSERT INTO FEINSUM_TIMING_FACTS"
        " (subscripts, index_to_length, use_matrix,"
        "  value_to_dtype, device_name, transform_id,"
        "  transform_params, runtime_in_sec,"
        "  compiler_version, giga_op_info, timestamp)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            subscripts,
            index_to_length,
            use_matrix,
            value_to_dtype,
            device_name,
            transform_id,
            transform_params,
            runtime_in_sec,
            compiler_version,
            giga_op_info,
            timestamp,
        ),
    )


def main():
    conn = sql.connect(
        "file:/home/line/projects/feinsum"
        "/data/transform_archive_v3.sqlite?mode=ro",
        uri=True,
    )
    cursor = conn.cursor()
    cursor.execute(
        "SELECT "
        " subscripts,"
        " index_to_length,"
        " use_matrix,"
        " value_to_dtype,"
        " device_name,"
        " transform_id,"
        " transform_params,"
        " runtime_in_sec,"
        " compiler_version,"
        " giga_op_info,"
        " timestamp"
        " FROM FEINSUM_TIMING_FACTS;"
    )
    all_timing_facts = cursor.fetchall()
    conn.close()

    conn = sql.connect("transform_archive_v4.sqlite")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE FEINSUM_TIMING_FACTS ("
        " ID INTEGER PRIMARY KEY AUTOINCREMENT,"
        " subscripts TEXT,"
        " index_to_length TEXT,"
        " use_matrix TEXT,"
        " value_to_dtype TEXT,"
        " device_name TEXT,"
        " transform_id TEXT,"
        " transform_params TEXT,"
        " runtime_in_sec REAL,"
        " compiler_version TEXT,"
        " giga_op_info TEXT,"
        " timestamp TEXT"
        ")"
    )

    print(len(all_timing_facts))
    for fact in all_timing_facts:
        (
            subscripts,
            idx_to_len,
            use_matrix,
            val_to_dtype,
            device_name,
            transform_id,
            transform_params,
            runtime_in_sec,
            compiler_version,
            giga_op_info,
            timestamp,
        ) = fact
        input_subscripts = subscripts.split("->")[0]
        idx_to_len = json.loads(idx_to_len)
        val_to_dtype = {
            val: np.dtype(dtype) for val, dtype in json.loads(val_to_dtype).items()
        }
        use_matrix = [
            [frozenset(uses) for uses in use_row]
            for use_row in json.loads(use_matrix)
        ]
        arg_shapes = [
            [idx_to_len.get(idx, np.inf) for idx in input_subscript]
            for input_subscript in input_subscripts.split(",")
        ]
        einsum = f.batched_einsum(
            subscripts, arg_shapes, use_matrix, value_to_dtype=val_to_dtype
        )
        record_into_db(
            cursor,
            einsum=einsum,
            device_name=device_name,
            transform_id=transform_id,
            transform_params=transform_params,
            runtime_in_sec=runtime_in_sec,
            compiler_version=compiler_version,
            giga_op_info=giga_op_info,
            timestamp=timestamp,
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
