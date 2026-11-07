from arguments import ConfigHandler


def test_bab_feasibility_tolerance_is_configurable_from_cli():
    handler = ConfigHandler()
    handler.parse_config(
        args=["--nlp_bab_feas_tol", "1e-6"],
        verbose=False,
    )

    assert handler["solver"]["nlp"]["bab_feas_tol"] == 1.0e-6
