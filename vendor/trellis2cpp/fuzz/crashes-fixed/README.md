# Fixed-crash regression seeds

Inputs that once crashed a fuzz harness and now must not. Re-run after changes:

    ./build-fuzz/fuzz/fuzz_dinodata fuzz/crashes-fixed/

- `dinodata-shape-overflow` — a `.dinodata` header whose shape dims multiply to
  an enormous element count; the loader used to `vector::resize` it and throw
  `std::length_error`. Fixed by an overflow-checked element-count cap in
  `trellis2_load_dinodata`.
