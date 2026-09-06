"""Canonical production schema for the mobile-cable ocean sensors."""

OCEAN_SENSOR_IDS = (
    "water_temperature_c",
    "water_depth_m",
    "turbidity_ntu",
    "significant_wave_height_m",
    "current_speed_mps",
)

OCEAN_SENSOR_UNITS = {
    "water_temperature_c": "degree_Celsius",
    "water_depth_m": "m",
    "turbidity_ntu": "NTU",
    "significant_wave_height_m": "m",
    "current_speed_mps": "m/s",
}
