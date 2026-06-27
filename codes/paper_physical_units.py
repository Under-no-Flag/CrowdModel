"""Physical-unit calibration for the paper figures.

The macroscopic solver runs in simulation units. For the paper we map those
units to physical (SI) units with three reference scales per scenario:

    * speed  : v_max = 1.5 sim-speed  ->  1.5 m/s  (free walking speed)
    * density: rho   -> ped/m^2       (rho_max is a realistic jam density)
    * length : kappa_L meters per simulation length unit (the one free scale)

The time scale then follows from the speed anchor, kappa_t = kappa_L, and the
mass scale from density x area, kappa_m = kappa_L^2.

Derived conversion factors (multiply a simulation value to get the SI value):

    length        x kappa_L                 -> m
    time          x kappa_t (= kappa_L)     -> s
    density       x kappa_rho (= 1)         -> ped/m^2
    mass / flow   x kappa_m (= kappa_L^2)   -> ped
    flow rate     x kappa_L                 -> ped/s   (mass/time)

Four-channel scenic-platform scenario
    Representative calibration (for readability; not a specific venue):
    grid cell dx_sim = 0.5 -> 1.0 m, so kappa_L = 2.0.

Bund-inspired scenario
    Georeferenced calibration. AnyLogic scene coordinates were matched to
    satellite latitude/longitude at four control points (anylogic-scene/
    BundScene/gis.json); a four-point affine fit gives 0.435 m per scene unit
    (sub-1.5 m residual). With 2053x899 scene units mapped onto the 512x224
    grid (4.01 units/cell), the physical cell size is ~1.745 m, so kappa_L
    = 1.745 / 0.125 = 13.96.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitCalibration:
    name: str
    dx_sim: float            # simulation grid spacing (length units per cell)
    m_per_cell: float        # measured/representative physical cell size (m)

    @property
    def kappa_L(self) -> float:
        """Meters per simulation length unit."""
        return self.m_per_cell / self.dx_sim

    @property
    def kappa_t(self) -> float:
        """Seconds per simulation time unit (speed anchored at 1.5 m/s)."""
        return self.kappa_L

    @property
    def ped_per_density_unit(self) -> float:
        return 1.0

    @property
    def ped_per_mass_unit(self) -> float:
        """ped per native solver mass unit (density x cell area)."""
        return self.kappa_L ** 2

    @property
    def ped_per_sec_per_rate_unit(self) -> float:
        """ped/s per simulation flow-rate unit (mass/time)."""
        return self.kappa_L


# dx_sim = 0.5, representative 1.0 m/cell  ->  kappa_L = 2.0
FOUR_CHANNEL = UnitCalibration(name="four_channel", dx_sim=0.5, m_per_cell=1.0)

# dx_sim = 0.125, georeferenced 1.745 m/cell  ->  kappa_L = 13.96
BUND = UnitCalibration(name="bund", dx_sim=0.125, m_per_cell=1.745)
