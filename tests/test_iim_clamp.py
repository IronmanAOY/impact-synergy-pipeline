import numpy as np
import pytest

from impact_pipeline import mpc_metrics as mm


def test_iim_canonical_mapping():
    ts = np.random.RandomState(0).rand(6, 60)
    info = mm.compute_IIM(
        ts,
        bins=3,
        n_parts=8,
        max_nodes=5,
        max_mechanism_size=2,
        max_purview_size=2,
        return_details=True,
    )
    assert info["defined"] is True
    assert info["canonical"] == pytest.approx(np.clip(info["raw"], 0.0, 1.0))
    assert 0.0 <= info["canonical"] <= 1.0

    raw_direct = mm.compute_IIM(
        ts,
        bins=3,
        n_parts=8,
        max_nodes=5,
        max_mechanism_size=2,
        max_purview_size=2,
        clamp=False,
        return_details=False,
    )
    can_direct = mm.compute_IIM(
        ts,
        bins=3,
        n_parts=8,
        max_nodes=5,
        max_mechanism_size=2,
        max_purview_size=2,
        clamp=True,
        return_details=False,
    )
    assert raw_direct == pytest.approx(info["raw"])
    assert can_direct == pytest.approx(info["canonical"])


def test_iim_exhaustive_defaults_and_cut_sampling():
    ts = np.random.RandomState(1).rand(4, 80)

    info_all = mm.compute_IIM(
        ts,
        bins=2,
        lag_trs=1,
        n_parts=None,  # exhaustive cuts
        max_nodes=4,
        max_mechanism_size=None,  # full mechanism order
        max_purview_size=None,    # full purview order
        return_details=True,
    )
    assert info_all["defined"] is True
    assert info_all["n_nodes_used"] == 4
    assert info_all["n_cuts_evaluated"] == 7  # 4 singletons + 3 balanced unique cuts
    assert info_all["max_mechanism_size_used"] == 4
    assert info_all["max_purview_size_used"] == 4

    info_sampled = mm.compute_IIM(
        ts,
        bins=2,
        lag_trs=1,
        n_parts=3,  # sampled subset of system cuts
        max_nodes=4,
        max_mechanism_size=2,
        max_purview_size=2,
        return_details=True,
    )
    assert info_sampled["defined"] is True
    assert info_sampled["n_cuts_evaluated"] == 3
