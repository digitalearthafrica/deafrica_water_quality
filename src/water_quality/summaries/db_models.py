from datetime import datetime

from sqlalchemy import Column, Date, Float, ForeignKey, String
from sqlalchemy.orm import Mapped
from waterbodies.db_models import WaterbodyBase


class WaterbodyWaterQuality(WaterbodyBase):
    __tablename__ = "waterbodies_water_quality"

    obs_id: Mapped[str] = Column(String, primary_key=True, index=True)
    uid: Mapped[str] = Column(
        String, ForeignKey("waterbodies_historical_extent.uid"), index=True
    )
    date: Mapped[datetime] = Column(Date, index=True)

    hue_q0_1: Mapped[float] = Column(Float, nullable=True)
    hue_q0_2: Mapped[float] = Column(Float, nullable=True)
    hue_q0_3: Mapped[float] = Column(Float, nullable=True)
    hue_q0_4: Mapped[float] = Column(Float, nullable=True)
    hue_q0_5: Mapped[float] = Column(Float, nullable=True)
    hue_q0_6: Mapped[float] = Column(Float, nullable=True)
    hue_q0_7: Mapped[float] = Column(Float, nullable=True)
    hue_q0_8: Mapped[float] = Column(Float, nullable=True)
    hue_q0_9: Mapped[float] = Column(Float, nullable=True)

    owt_q0_1: Mapped[float] = Column(Float, nullable=True)
    owt_q0_2: Mapped[float] = Column(Float, nullable=True)
    owt_q0_3: Mapped[float] = Column(Float, nullable=True)
    owt_q0_4: Mapped[float] = Column(Float, nullable=True)
    owt_q0_5: Mapped[float] = Column(Float, nullable=True)
    owt_q0_6: Mapped[float] = Column(Float, nullable=True)
    owt_q0_7: Mapped[float] = Column(Float, nullable=True)
    owt_q0_8: Mapped[float] = Column(Float, nullable=True)
    owt_q0_9: Mapped[float] = Column(Float, nullable=True)

    chla_q0_1: Mapped[float] = Column(Float, nullable=True)
    chla_q0_2: Mapped[float] = Column(Float, nullable=True)
    chla_q0_3: Mapped[float] = Column(Float, nullable=True)
    chla_q0_4: Mapped[float] = Column(Float, nullable=True)
    chla_q0_5: Mapped[float] = Column(Float, nullable=True)
    chla_q0_6: Mapped[float] = Column(Float, nullable=True)
    chla_q0_7: Mapped[float] = Column(Float, nullable=True)
    chla_q0_8: Mapped[float] = Column(Float, nullable=True)
    chla_q0_9: Mapped[float] = Column(Float, nullable=True)

    tsi_q0_1: Mapped[float] = Column(Float, nullable=True)
    tsi_q0_2: Mapped[float] = Column(Float, nullable=True)
    tsi_q0_3: Mapped[float] = Column(Float, nullable=True)
    tsi_q0_4: Mapped[float] = Column(Float, nullable=True)
    tsi_q0_5: Mapped[float] = Column(Float, nullable=True)
    tsi_q0_6: Mapped[float] = Column(Float, nullable=True)
    tsi_q0_7: Mapped[float] = Column(Float, nullable=True)
    tsi_q0_8: Mapped[float] = Column(Float, nullable=True)
    tsi_q0_9: Mapped[float] = Column(Float, nullable=True)

    tsm_q0_1: Mapped[float] = Column(Float, nullable=True)
    tsm_q0_2: Mapped[float] = Column(Float, nullable=True)
    tsm_q0_3: Mapped[float] = Column(Float, nullable=True)
    tsm_q0_4: Mapped[float] = Column(Float, nullable=True)
    tsm_q0_5: Mapped[float] = Column(Float, nullable=True)
    tsm_q0_6: Mapped[float] = Column(Float, nullable=True)
    tsm_q0_7: Mapped[float] = Column(Float, nullable=True)
    tsm_q0_8: Mapped[float] = Column(Float, nullable=True)
    tsm_q0_9: Mapped[float] = Column(Float, nullable=True)

    st_max_q0_1: Mapped[float] = Column(Float, nullable=True)
    st_max_q0_2: Mapped[float] = Column(Float, nullable=True)
    st_max_q0_3: Mapped[float] = Column(Float, nullable=True)
    st_max_q0_4: Mapped[float] = Column(Float, nullable=True)
    st_max_q0_5: Mapped[float] = Column(Float, nullable=True)
    st_max_q0_6: Mapped[float] = Column(Float, nullable=True)
    st_max_q0_7: Mapped[float] = Column(Float, nullable=True)
    st_max_q0_8: Mapped[float] = Column(Float, nullable=True)
    st_max_q0_9: Mapped[float] = Column(Float, nullable=True)

    st_median_q0_1: Mapped[float] = Column(Float, nullable=True)
    st_median_q0_2: Mapped[float] = Column(Float, nullable=True)
    st_median_q0_3: Mapped[float] = Column(Float, nullable=True)
    st_median_q0_4: Mapped[float] = Column(Float, nullable=True)
    st_median_q0_5: Mapped[float] = Column(Float, nullable=True)
    st_median_q0_6: Mapped[float] = Column(Float, nullable=True)
    st_median_q0_7: Mapped[float] = Column(Float, nullable=True)
    st_median_q0_8: Mapped[float] = Column(Float, nullable=True)
    st_median_q0_9: Mapped[float] = Column(Float, nullable=True)

    st_min_q0_1: Mapped[float] = Column(Float, nullable=True)
    st_min_q0_2: Mapped[float] = Column(Float, nullable=True)
    st_min_q0_3: Mapped[float] = Column(Float, nullable=True)
    st_min_q0_4: Mapped[float] = Column(Float, nullable=True)
    st_min_q0_5: Mapped[float] = Column(Float, nullable=True)
    st_min_q0_6: Mapped[float] = Column(Float, nullable=True)
    st_min_q0_7: Mapped[float] = Column(Float, nullable=True)
    st_min_q0_8: Mapped[float] = Column(Float, nullable=True)
    st_min_q0_9: Mapped[float] = Column(Float, nullable=True)

    fai_cover: Mapped[float] = Column(Float, nullable=True)
    ndvi_cover: Mapped[float] = Column(Float, nullable=True)
    water_area_m2: Mapped[float] = Column(Float, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<WaterbodyWaterQuality obs_id={self.obs_id}, uid={self.uid}, "
            + f"date={self.date}, ...>"
        )


class WaterbodyWaterQualityPercentiles(WaterbodyBase):
    __tablename__ = "waterbodies_water_quality_percentiles"

    uid: Mapped[str] = Column(
        String,
        ForeignKey("waterbodies_historical_extent.uid"),
        primary_key=True,
        index=True,
    )
    hue_q0_5_percentile: Mapped[float] = Column(Float, nullable=True)
    owt_q0_5_percentile: Mapped[float] = Column(Float, nullable=True)
    chla_q0_5_percentile: Mapped[float] = Column(Float, nullable=True)
    tsi_q0_5_percentile: Mapped[float] = Column(Float, nullable=True)
    tsm_q0_5_percentile: Mapped[float] = Column(Float, nullable=True)
    st_max_q0_5_percentile: Mapped[float] = Column(Float, nullable=True)
    st_median_q0_5_percentile: Mapped[float] = Column(Float, nullable=True)
    st_min_q0_5_percentile: Mapped[float] = Column(Float, nullable=True)
    fai_cover_percentile: Mapped[float] = Column(Float, nullable=True)
    ndvi_cover_percentile: Mapped[float] = Column(Float, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<WaterbodyWaterQualityPercentiles uid={self.uid}, "
            + f"hue_q0_5_percentile={self.hue_q0_5_percentile}, ...>"
        )
