from water_quality.grid import check_resolution


def check_config(analysis_config: dict) -> dict:
    """Validate the provided analysis configuration dictionary and
    extract the necessary parameters.

    This function ensures that the `analysis_config` is not None and
    contains all the necessary parameters required for the analysis.
    If any essential parameters are missing or the config itself is
    empty, a ValueError is raised.

    Parameters
    ----------
    analysis_config : dict
        A dictionary containing the configuration parameters for the
        analysis.

    Returns
    -------
    dict
        The validated `analysis_config` dictionary if all checks pass.
    """

    if analysis_config is None:
        raise ValueError(
            "Please provide a config for the analysis parameters in "
            "yaml format, file or text"
        )

    config_items = [
        "resolution",
        "instruments_to_use",
        "water_frequency_threshold_high",
        "water_frequency_threshold_low",
        "permanent_water_threshold",
        "sigma_coefficient",
        "product",
    ]

    missing_parameters = []
    for k in config_items:
        if k not in list(analysis_config.keys()):
            missing_parameters.append(k)
    if missing_parameters:
        raise ValueError(
            "The following analysis parameters not found "
            f"{', '.join(missing_parameters)} "
        )

    resolution_m = check_resolution(int(analysis_config["resolution"]))
    product_info = analysis_config["product"]

    return dict(
        resolution=resolution_m,
        WFTH=analysis_config["water_frequency_threshold_high"],
        WFTL=analysis_config["water_frequency_threshold_low"],
        PWT=analysis_config["permanent_water_threshold"],
        SC=analysis_config["sigma_coefficient"],
        product_name=product_info["name"],
        product_version=product_info["version"],
        instruments_to_use=analysis_config["instruments_to_use"],
    )
