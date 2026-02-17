from odc.geo import XY, Resolution
from odc.geo.crs import CRS
from odc.geo.gridspec import GridSpec


def check_crs(crs: str | CRS) -> CRS:
    """
    Ensure crs provided is a projected crs with dimension units
    in meters.

    Parameters
    ----------
    crs : str | CRS
        CRS to check.

    Returns
    -------
    CRS
        Input crs if the CRS has passed all checks.
    """
    if isinstance(crs, str):
        crs = CRS(crs)

    # Checks if the CRS is projected
    try:
        assert crs.projected
    except AssertionError:
        raise ValueError(f"{crs} is not a projected CRS")
    # Checks if the dimension units of the crs is in meters
    try:
        assert "metre" in crs.units
    except AssertionError:
        raise ValueError(f"{crs} dimension units are not metres")
    return crs


def check_resolution(resolution: int) -> int:
    """
    Ensure resolution provided is a positive integer.

    Parameters
    ----------
    resolution : int
        Resolution provided.

    Returns
    -------
    int
        Input resolution if the value has passed all checks
    """
    if resolution < 0:
        raise ValueError(
            f"Expecting positive value for resolution not {resolution}"
        )
    if not isinstance(resolution, int):
        raise ValueError(
            f"Expecting resolution to be an integer not {type(resolution)}"
        )
    return resolution


def get_waterbodies_grid(resolution_m: int = 10) -> GridSpec:
    """
    Get the spatial grid to be used for tiling in the DE Africa
    Water Quality continental workflow. The resulting tiles
    cover the same tiles (regions) as the DE Africa GeoMAD products.

    Parameters
    ----------
    resolution_m : int
        Pixel resolution in meters.

    Returns
    -------
    GridSpec
        Gridspec to be used for tiling.
    """
    crs = check_crs("EPSG:6933")
    resolution_m = check_resolution(resolution_m)

    # Size of each tile in meters (CRS units).
    # To match the tile size for DE Africa GeoMAD products.
    tile_size = 96000

    # Size of each tile in pixels.
    # This ensures no matter the resolution the tile extents
    # (geometries) will remain the same.
    tile_shape = tile_size / resolution_m

    gridspec = GridSpec(
        crs=crs,
        tile_shape=XY(
            y=tile_shape,
            x=tile_shape,
        ),
        resolution=Resolution(y=-resolution_m, x=resolution_m),
        origin=XY(y=-7392000, x=-17376000),
    )
    return gridspec
