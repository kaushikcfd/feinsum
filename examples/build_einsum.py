import feinsum as f

print(
    f.einsum(
        "ij,j->i",
        f.array((10, 4), "float32"),
        f.array((4,), "float32"),
    )
)
