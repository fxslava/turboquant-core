# Review 1: scalar ownership and uniform source order

The successor address is computed after phase4, retained in scalar state through
phases5..7, and installed only for the next strip. First metadata is acquired
before initial quantization. Exactly N mapping reads for N strips; no out-of-range
successor access. CPU address-lifetime model passed strip counts1..129. The hot
grinder has no added runtime branch; terminal work remains separately peeled.
Baseline cold GM dependencies remain explicit and accounted under the user's
clarified ablation rule. No claim of zero total scalar stalls is made.
