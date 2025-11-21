from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from water_quality.io import join_url
from water_quality.logs import setup_logging


def place_to_parquet():
    log = setup_logging()
    data_dir = Path(__file__).resolve().parent
    output_file = join_url(str(data_dir), "places.parquet")

    year1, year2 = 2000, 2000
    places = {
        "Lake_Baringo": {
            "run": True,
            "xyt": {
                "x": (36.00, 36.17),
                "y": (00.45, 00.74),
                "time": (year1, year2),
            },
            "desc": "Lake Baringo",
        },
        "Lake_Tikpan": {
            "run": True,
            "xyt": {
                "x": (1.8215, 1.8265),
                "y": (6.459, 6.4626),
                "time": (year1, year2),
            },
            "desc": "Lake Tikpan",
        },
        "Lake_Chad": {
            "run": True,
            "xyt": {
                "x": (12.97, 15.50),
                "y": (12.40, 14.50),
                "time": (year1, year2),
            },
            "desc": "Lake Chad",
        },
        "Weija_Reservoir": {
            "run": True,
            "xyt": {
                "x": (-0.325, -0.41),
                "y": (5.54, 5.62),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "Senegal_StLouis": {
            "run": True,
            "xyt": {
                "x": (-15.74, -15.84),
                "y": (16.3, 16.3900),
                "time": (year1, year2),
            },
            "desc": "Lac de Guiers",
        },
        "Lake_Sulunga": {
            "run": True,
            "xyt": {
                "x": (34.95, 35.4),
                "y": (-6.3, -5.8),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "few_pixels": {
            "run": False,
            "xyt": {
                "x": (33.1600, 33.16005),
                "y": (-2.1200, -2.1424),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "small_area": {
            "run": False,
            "xyt": {
                "x": (33.1655, 33.1864),
                "y": (-2.1532, -2.1444),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "cameroon_res1": {
            "run": True,
            "xyt": {
                "y": (6.20, 6.30),
                "x": (11.25, 11.35),
                "time": (year1, year2),
            },
            "desc": "reservoir in cameroon",
        },
        "Lake_Victoria": {
            "run": True,
            "xyt": {
                "x": (33.100, 33.300),
                "y": (-2.0800, -2.1500),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "Lake_Mweru": {
            "run": True,
            "xyt": {
                "x": (28.200, 29.300),
                "y": (-9.500, -8.500),
                "time": (year1, year2),
            },
            "desc": "Lake Mweru - Zambia / DRC",
        },
        "Lake_Mweru_subset": {
            "run": True,
            "xyt": {
                "x": (28.450, 28.750),
                "y": (-9.180, -9.030),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "Ghana_AwunaBeach": {
            "run": True,
            "xyt": {
                "x": (-1.580, -1.640),
                "y": (5.0, 5.05),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "Ghana_River": {
            "run": True,
            "xyt": {
                "x": (-1.626, -1.610),
                "y": (5.065, 5.089),
                "time": (year1, year2),
            },
            "desc": "Ghana, turbid river",
        },
        "Large_area": {
            "run": False,
            "xyt": {
                "x": (32.5, 35.5),
                "y": (-4.5, -1.5),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "Lake_vic_west": {
            "run": True,
            "xyt": {
                "x": (32.5, 32.78),
                "y": (-2.65, -2.3),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "Lake_vic_east": {
            "run": True,
            "xyt": {
                "x": (32.78, 33.3),
                "y": (-2.65, -2.3),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "Lake_vic_test": {
            "run": True,
            "xyt": {
                "x": (32.78, 33.13),
                "y": (-1.95, -1.6),
                "time": (year1, year2),
            },
            "desc": "Lake Victoria cloud affected",
        },
        "Lake_vic_turbid": {
            "run": True,
            "xyt": {
                "x": (34.60, 34.70),
                "y": (-0.25, -0.20),
                "time": (year1, year2),
            },
            "desc": "Lake Victoria turbid area in NE",
        },
        "Lake_vic_algae": {
            "run": True,
            "xyt": {
                "x": (34.62, 34.78),
                "y": (-0.18, -0.08),
                "time": (year1, year2),
            },
            "desc": "Lake Victoria Water Hyacinth affected area in NE, port Kisumu",
        },
        "Lake_vic_clear": {
            "run": True,
            "xyt": {
                "x": (34.00, 34.10),
                "y": (-0.32, -0.27),
                "time": (year1, year2),
            },
            "desc": "Lake Victoria clear water area",
        },
        "Lake_Victoria_NE": {
            "run": True,
            "xyt": {
                "x": (33.5, 34.8),
                "y": (-0.6, 0.4),
                "time": (year1, year2),
            },
            "desc": "Lake Victoria NE",
        },
        "Morocco": {
            "run": True,
            "xyt": {
                "x": (-7.45, -7.65),
                "y": (32.4, 32.5),
                "time": (year1, year2),
            },
            "desc": "Barrage Al Massira",
        },
        "Thewaterskool_SA": {
            "run": True,
            "xyt": {
                "x": (19.1, 19.3),
                "y": (-34.1, -33.98),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "SA_dam": {
            "run": True,
            "xyt": {
                "x": (19.35, 19.47),
                "y": (-33.800, -33.650),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "SA_dam_north": {
            "run": True,
            "xyt": {
                "x": (19.42, 19.44),
                "y": (-33.73, -33.699),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "SA_dam_south": {
            "run": True,
            "xyt": {
                "x": (19.415, 19.431),
                "y": (-33.781, -33.772),
                "time": (year1, year2),
            },
            "desc": "",
        },
        "Ethiopia1of2": {
            "run": True,
            "xyt": {
                "x": (38.35, 38.65),
                "y": (7.37, 7.55),
                "time": (year1, year2),
            },
            "desc": "Ethiopia, Shala Hayk'",
        },
        "Ethiopia2of2": {
            "run": True,
            "xyt": {
                "x": (38.66, 38.83),
                "y": (7.50, 7.71),
                "time": (year1, year2),
            },
            "desc": "Ethiopia, Abyata Hayk'",
        },
        "Ethiopia3of2": {
            "run": True,
            "xyt": {
                "x": (38.50, 38.67),
                "y": (7.55, 7.69),
                "time": (year1, year2),
            },
            "desc": "Ethiopia, Langano Hayk' (turbid)",
        },
        "Ethiopia_Lake_Tana": {
            "run": True,
            "xyt": {
                "x": (37.05, 37.22),
                "y": (11.9, 12.0),
                "time": (year1, year2),
            },
            "desc": "Ethiopia_Lake_Tana",
        },
        "Mare_aux_Vacoas": {
            "run": True,
            "xyt": {
                "x": (57.485, 57.524),
                "y": (-20.389, -20.359),
                "time": (year1, year2),
            },
            "desc": "Mare_aux_Vacoas",
        },
        "SA_smalldam": {
            "run": True,
            "xyt": {
                "x": (19.494, 19.498),
                "y": (-33.802, -33.800),
                "time": (year1, year2),
            },
            "desc": "Irrigation Dam, South Africa",
        },
        "SA_smalldam1": {
            "run": True,
            "xyt": {
                "x": (19.505, 19.510),
                "y": (-33.8065, -33.803),
                "time": (year1, year2),
            },
            "desc": "Irrigation Dam, South Africa, clear water",
        },
        "Ethiopia_both": {
            "run": False,
            "xyt": {
                "x": (38.35, 38.83),
                "y": (7.37, 7.71),
                "time": (year1, year2),
            },
            "desc": "Ethiopia, Lake Abiata +",
        },
        "Lake Chamo": {
            "run": True,
            "xyt": {
                "x": (37.45, 37.65),
                "y": (5.685, 5.979),
                "time": (year1, year2),
            },
            "desc": "Lake Chamo, Ethiopia",
        },
        "Lake Ziway": {
            "run": True,
            "xyt": {
                "x": (38.711, 38.966),
                "y": (7.838, 8.148),
                "time": (year1, year2),
            },
            "desc": "Lake Ziway, Ethiopia",
        },
        "Lake Alwassa": {
            "run": True,
            "xyt": {
                "x": (38.380, 38.493),
                "y": (6.977, 7.133),
                "time": (year1, year2),
            },
            "desc": "Lake Alwassa, Ethiopia",
        },
        "Lake Elmenteita": {
            "run": True,
            "xyt": {
                "x": (36.211, 36.273),
                "y": (-0.488, -0.401),
                "time": (year1, year2),
            },
            "desc": "Lake Elmenteita, Kenya",
        },
        "Madagascar": {
            "run": True,
            "xyt": {
                "x": (43.58, 43.76),
                "y": (-22.03, -21.87),
                "time": (year1, year2),
            },
            "desc": "Farihy Ihotry, Madagascar",
        },
        "Lake_Manyara": {
            "run": True,
            "xyt": {
                "x": (35.724, 35.929),
                "y": (-03.814, -03.409),
                "time": (year1, year2),
            },
            "desc": "Lake_Manyara, Tanzania",
        },  # this is the lake to use as an example of monitoring, see 2015-12-28
        "Farihy_": {
            "run": True,
            "xyt": {
                "x": (43.58, 43.76),
                "y": (-22.03, -21.87),
                "time": (year1, year2),
            },
            "desc": "Farihy Ihotry, Madagascar",
        },
        "Farihy_itasy": {
            "run": True,
            "xyt": {
                "x": (46.73, 46.83),
                "y": (-19.10, -19.04),
                "time": (year1, year2),
            },
            "desc": "Farihy Itasy, Madagascar",
        },
        "Kolokonda": {
            "run": True,
            "xyt": {
                "x": (35.4888, 35.5488),
                "y": (-5.976, -5.916),
                "time": (year1, year2),
            },
            "desc": "Kolokonda, Tanzania",
        },
        "Dodoma_small": {
            "run": True,
            "xyt": {
                "x": (35.475, 35.51),
                "y": (-6.03, -5.99),
                "time": (year1, year2),
            },
            "desc": "Dodoma, Tanzania",
        },
        "size_test": {
            "run": False,
            "xyt": {
                "x": (31.400, 32.40),
                "y": (-0.00, -1.00),
                "time": (year1, year2),
            },
            "desc": "Lake Victoria",
        },
        "lake_vic_all": {
            "run": False,
            "xyt": {
                "x": (31.500, 34.86),
                "y": (-3.00, +0.50),
                "time": (year1, year2),
            },
            "desc": "Lake Victoria",
        },
        "lake_elmenteita": {
            "run": True,
            "xyt": {
                "x": (36.200, 36.27),
                "y": (-0.485, -0.390),
                "time": (year1, year2),
            },
            "desc": "Lake Elmenteita",
        },
        "mombasa": {
            "run": True,
            "xyt": {
                "x": (39.500, 39.72),
                "y": (-4.10, -3.97),
                "time": (year1, year2),
            },
            "desc": "Mombasa",
        },
        "Mauritania_2": {
            "run": True,
            "xyt": {
                "x": (-15.63, -15.54),
                "y": (16.605, 16.69),
                "time": (year1, year2),
            },
            "desc": "Mauritania Wetland",
        },
        "Mauritania_1": {
            "run": True,
            "xyt": {
                "x": (-16.37, -16.32),
                "y": (16.41, 16.45),
                "time": (year1, year2),
            },
            "desc": "Mauritania Wetland",
        },
        "Lake_Nasser_nth": {
            "run": True,
            "xyt": {
                "x": (32.87, 32.95),
                "y": (23.69, 23.72),
                "time": (year1, year2),
            },
            "desc": "Lake Nasser clear water",
        },
        "Lake_Nasser_sth": {
            "run": True,
            "xyt": {
                "x": (31.20, 31.30),
                "y": (21.795, 21.845),
                "time": (year1, year2),
            },
            "desc": "Lake Nasser turbid water",
        },
        "Tana_Hayk": {
            "run": True,
            "xyt": {
                "x": (36.95, 37.65),
                "y": (11.56, 12.33),
                "time": (year1, year2),
            },
            "desc": "T'ana Hayk', northern Ethiopia",
        },
        "Lake_Malawi": {
            "run": True,
            "xyt": {
                "x": (34.25, 34.97),
                "y": (-13.6, -13.3),
                "time": (year1, year2),
            },
            "desc": "Lake Malawi - part of",
        },
        "Lago de Cabora": {
            "run": True,
            "xyt": {
                "x": (30.90, 32.52),
                "y": (-15.95, -15.45),
                "time": (year1, year2),
            },
            "desc": "Lago de Cabora Basa - Mozambique",
        },
        "Mtera Reservoir": {
            "run": True,
            "xyt": {
                "x": (35.60, 36.01),
                "y": (-7.20, -6.86),
                "time": (year1, year2),
            },
            "desc": "Lake Nzuhe, Tanzania",
        },
        "Barrage Joumine": {
            "run": True,
            "xyt": {
                "x": (09.53, 09.62),
                "y": (36.952, 37.00),
                "time": (year1, year2),
            },
            "desc": "Joumine Dam,Tunisia",
        },
        "Tunisia_Dam": {
            "run": True,
            "xyt": {
                "x": (08.53, 08.56),
                "y": (36.685, 36.75),
                "time": (year1, year2),
            },
            "desc": "Tunisia",
        },
        "Lake_Ngami": {
            "run": True,
            "xyt": {
                "x": (22.55, 22.89),
                "y": (-20.6, -20.37),
                "time": (year1, year2),
            },
            "desc": "Botswana",
        },
        "Lake_Chilwa": {
            "run": True,
            "xyt": {
                "x": (35.5, 35.9),
                "y": (-15.6, -14.90),
                "time": (year1, year2),
            },
            "desc": "Malawi - Lake Chilwa",
        },
        "Lake_Malombe": {
            "run": True,
            "xyt": {
                "x": (35.15, 35.35),
                "y": (-14.8, -14.50),
                "time": (year1, year2),
            },
            "desc": "Malawi - Lake Malombe",
        },
        "Lake_Piti": {
            "run": True,
            "xyt": {
                "x": (32.85, 32.90),
                "y": (-26.6, -26.50),
                "time": (year1, year2),
            },
            "desc": "Mozambique - Lake Piti",
        },
        "Maputo_reserve": {
            "run": True,
            "xyt": {
                "x": (32.79, 32.83),
                "y": (-26.55, -26.50),
                "time": (year1, year2),
            },
            "desc": "Mozambique - Maputo reserve",
        },
        "Indian_Ocean": {
            "run": True,
            "xyt": {
                "x": (57.75, 57.80),
                "y": (-20.5, -20.45),
                "time": (year1, year2),
            },
            "desc": "Mauritius - Oceanic waters",
        },
        "Mare_Vacoas": {
            "run": True,
            "xyt": {
                "x": (57.48, 57.52),
                "y": (-20.38, -20.36),
                "time": (year1, year2),
            },
            "desc": "Mauritius - Mare aux Vacoas",
        },
        "Naute": {
            "run": True,
            "xyt": {
                "x": (17.93, 18.05),
                "y": (-26.97, -26.92),
                "time": (year1, year2),
            },
            "desc": "Namibia - Naute reserve",
        },
        "Lake_Turkana": {
            "run": True,
            "xyt": {
                "x": (35.80, 36.72),
                "y": (2.38, 4.79),
                "time": (year1, year2),
            },
            "desc": "Kenya -- Lake Turkana",
        },
        "Haartbeesport_dam": {
            "run": True,
            "xyt": {
                "x": (27.7972, 27.91117),
                "y": (-25.7761, -25.7275),
                "time": (year1, year2),
            },
            "desc": "Haartbeesport Dam  -- South Africa",
        },
        "Lake Bogoria": {
            "run": True,
            "xyt": {
                "x": (36.058, 36.133),
                "y": (0.1791, 0.3534),
                "time": (year1, year2),
            },
            "desc": "Lake Bogoria -- Tanzania",
        },
    }

    data = []
    for place_name in list(places.keys()):
        name = place_name
        description = places[place_name].get("desc", "")
        x = places[name]["xyt"]["x"]
        y = places[name]["xyt"]["y"]

        item = {
            "name": name,
            "description": description,
            "geometry": box(
                minx=min(x), maxx=max(x), miny=min(y), maxy=max(y)
            ),
        }
        data.append(item)

    gdf = gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326")
    gdf.to_parquet(output_file)
    log.info(f"Places table written to {output_file}")


if __name__ == "__main__":
    place_to_parquet()
