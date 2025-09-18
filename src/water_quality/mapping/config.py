def check_config(analysis_config: dict):
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
    ]
    missing_parameters = []
    for k in config_items:
        if k not in list(analysis_config.keys()):
            missing_parameters.append(k)
    if missing_parameters:
        raise ValueError(
            f"The following analysis parameters not found {', '.join(missing_parameters)} "
        )

    return analysis_config
