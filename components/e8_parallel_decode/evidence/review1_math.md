# Independent mathematical review of parallel decoder

PASS for reports/checkpoint2_20261123/parallel_decode/e8_parallel_decode.hpp,
SHA a501bb69f4443ca339bdd20f31b63d5f819d322f969bf16f495ca9551c481777.
Reviewed against frozen decoder SHA28f13039c8fec29ef91919a69a8314a60c9cb9aec3f5f3fc75b270f8e524cda2
and frozen e8_reference.py SHAa5b6c0d0b9d32ed53f285c7dbbbe63dbd8e5464019ebe461bb4d0b9f967ad495.

The eight Project expressions are the exact rows of the frozen integer G2
mapping. RepairKey orders first by absolute residual, then by nonnegative sign,
then by earliest index for nonnegative or latest index for negative. This is
precisely the old scan's tie behavior: first positive maximum if one exists,
otherwise the last negative maximum. Parity inputs are0/1 from balanced XOR,
so the masked repair changes exactly one coordinate by±8 or changes none.

Unsigned modulo arithmetic for p+4 preserves the low modulo8 residue and
floor-division parity even for negative projected values. The half-coset parity
and residual expressions are the same algebra as the frozen bit routine.
After repair, half-vs-integer coordinate0 residuals differ by4mod8, because
repairs add multiples8. They cannot be equal, so comparing coordinate0 on a
distance tie implements full lexicographic residual ordering exactly. Sum8 and
Xor8 are balanced exact integer reductions; squared-distance sums are<=512,
well within int32.

No mathematical discrepancy was found. The owner's exhaustive65536-label and
scale controls supplement this review. This signoff does not certify compiled
branchlessness, register/spill resources, UB layout, execution or timing. Actual
ternary/select lowering and device parity still require the coordinated gates.
