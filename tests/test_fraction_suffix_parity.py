from __future__ import annotations

"""GPU/CPU strategy-suffix parser parity (issue #21/#8 divergence fix).

The GPU helper (hf_paged_pq_intervention_joint_one_group._fraction_suffix)
historically split at the FIRST occurrence of the marker character, so
proxy_mass_m0p9 matched the "m" of "proxy_Mass" and silently fell back to
the default (mass 0.5 -> floor-clamped starts) while the CPU runner
(run_joint_kv_budget_policy_eval._parse_fraction_suffix) parsed 0.9 with a
boundary-anchored regex. This test pins both parsers to identical outputs
on the strategy strings in use, so the divergence cannot silently return.
"""

import pytest

from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_one_group import (
    _fraction_suffix as gpu_fraction_suffix,
)
from benchmark.selector_eval.runners.run_joint_kv_budget_policy_eval import (
    _parse_fraction_suffix as cpu_fraction_suffix,
)

# (strategy string, marker, default, expected)
CASES = [
    ("proxy_mass_m0p9", "m", 0.5, 0.9),
    ("proxy_mass_m0p8", "m", 0.5, 0.8),
    ("proxy_mass_m0p5", "m", 0.5, 0.5),
    ("proxy_mass_m0p35", "m", 0.5, 0.35),
    ("fixed_f0p05", "f", 0.05, 0.05),
    ("fixed_f0p1", "f", 0.05, 0.1),
    ("proxy_entropy_f0p25", "f", 0.05, 0.25),
    # No parsable suffix -> default (both sides).
    ("min", "m", 0.5, 0.5),
    ("zero", "m", 0.5, 0.5),
    ("proxy_mass_m", "m", 0.5, 0.5),
]


@pytest.mark.parametrize("name,marker,default,expected", CASES)
def test_gpu_parser_expected(name: str, marker: str, default: float, expected: float) -> None:
    assert gpu_fraction_suffix(name, marker, default) == pytest.approx(expected)


@pytest.mark.parametrize("name,marker,default,expected", CASES)
def test_gpu_cpu_parity(name: str, marker: str, default: float, expected: float) -> None:
    assert gpu_fraction_suffix(name, marker, default) == pytest.approx(
        cpu_fraction_suffix(name, marker, default)
    )
