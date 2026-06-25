from impact_pipeline import mpc_metrics as mm


def test_ci_zero_when_iim_undefined():
    out = mm.compute_CI(
        ram=1.0,
        pdi=1.0,
        nas=1.0,
        iim=float("nan"),
        srpi=1.0,
        defined={"IIM": False},
        return_details=True,
    )
    assert out["value"] == 0.0
    assert "IIM" in out["undefined_weighted_components"]
