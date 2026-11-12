"""
Iridology organ zone definitions (left iris primary, per Punia et al.).

Zones are sectors defined by clock-hour angles and radial bands
as fractions of the detected iris radius.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class IrisZone:
    name: str
    label: str
    hour_start: float  # clock position on iris chart (12 = top)
    hour_end: float
    radius_inner: float  # fraction of iris radius from pupil center
    radius_outer: float
    condition: str


def hour_to_degrees(hour: float) -> float:
    """Convert iris clock hour to degrees (0° = 3 o'clock, clockwise)."""
    return (hour - 3.0) * 30.0


# Organ zones mapped from Punia et al. + iridology charts (Bansal, Songire)
SYSTEMIC_ZONES: dict[str, IrisZone] = {
    "heart": IrisZone(
        name="heart",
        label="Heart / Cardiovascular",
        hour_start=2.2,
        hour_end=3.1,
        radius_inner=0.40,
        radius_outer=0.58,
        condition="heart_disease",
    ),
    "pancreas": IrisZone(
        name="pancreas",
        label="Pancreas / Diabetes",
        hour_start=3.5,
        hour_end=4.2,
        radius_inner=0.55,
        radius_outer=0.72,
        condition="diabetes",
    ),
    "cholesterol": IrisZone(
        name="cholesterol",
        label="Cholesterol / Lipid ring",
        hour_start=5.0,
        hour_end=7.0,
        radius_inner=0.68,
        radius_outer=0.92,
        condition="cholesterol",
    ),
    "lung": IrisZone(
        name="lung",
        label="Lung / Respiratory",
        hour_start=9.0,
        hour_end=11.0,
        radius_inner=0.45,
        radius_outer=0.75,
        condition="lung_disease",
    ),
}
