from typing import Dict, List, Union

from ... import weather as rk_weather
from .solar_workflow_manager import SolarWorkflowManager


def openfield_pv_merra_ryberg2019(
    placements,
    merra_path,
    global_solar_atlas_ghi_path,
    module="WINAICO WSx-240P6",
    elev=300,
    tracking="fixed",
    inverter=None,
    inverter_kwargs={},
    tracking_args={},
    output_netcdf_path=None,
    output_variables=None,
):
    """

    openfield_pv_merra_ryberg2019(placements, merra_path, global_solar_atlas_ghi_path, module="WINAICO WSx-240P6", elev=300, tracking="fixed",
                                    inverter=None, inverter_kwargs={}, tracking_args={}, output_netcdf_path=None, output_variables=None)

    Simulation of an openfield  PV openfield system based on MERRA Data.

    Parameters
    ----------
    placements: Pandas Dataframe
        Locations where to perform simulations at.
        Columns need to be lat (latitudes), lon (longitudes), tilt and capacity.

    merra_path: str
        Path to the MERRA Data on your computer.
        Can be a single ".nc" file, or a directory containing many ".nc" files.

    global_solar_atlas_ghi_path: str
        Path to the global solar atlas ghi data on your computer.

    module: str
        Name of the module that you want to use for the simulation.
        Default is Winaico Wsx-240P6.
        See reskit.solar.SolarWorkflowManager.configure_cec_module for more usage information.

    elev: float
        Elevation that you want to model your PV system at.

    tracking: str
                Option 1 is 'fixed' meaning that the module does not have any tracking capabilities.
                Option 2 is 'single-axis' meaning that the module has single-axis tracking capabilities.


    inverter: str
        Determines wether or not you want to model your PV system with an inverter.
        Default is None, meaning no inverter is assumed
        See reskit.solar.SolarWorkflowManager.apply_inverter_losses for more usage information

    output_netcdf_path: str
        Path to a file that you want to save your output NETCDF file at.
        Default is None

    output_variables: str
        Output variables of the simulation that you want to save into your NETCDF Outputfile.


    Returns
    -------
    A xarray dataset including all the output variables you defined as your output_variables.

    """

    wf = SolarWorkflowManager(placements)
    wf.configure_cec_module(module)

    if not "tilt" in wf.placements.columns:
        wf.estimate_tilt_from_latitude(convention="Ryberg2020")
    if not "azimuth" in wf.placements.columns:
        wf.estimate_azimuth_from_latitude()
    if not "elev" in wf.placements.columns:
        wf.apply_elevation(elev)

    wf.read(
        variables=[
            "surface_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
            "surface_dew_temperature",
            "global_horizontal_irradiance",
        ],
        source_type="MERRA",
        source=merra_path,
        set_time_index=True,
        verbose=False,
    )

    wf.adjust_variable_to_long_run_average(
        variable="global_horizontal_irradiance",
        source_long_run_average=rk_weather.MerraSource.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=global_solar_atlas_ghi_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )

    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()
    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370)
    wf.determine_air_mass(model="kastenyoung1989")
    wf.apply_DIRINT_model()
    wf.diffuse_horizontal_irradiance_from_trigonometry()

    if tracking == "single_axis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sapm()

    wf.simulate_with_interpolated_single_diode_approximation(module=module)

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    wf.apply_loss_factor(0.20, variables=["capacity_factor", "total_system_generation"])

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def openfield_pv_era5_unvalidated(
    placements,
    era5_path,
    global_solar_atlas_ghi_path,
    global_solar_atlas_dni_path,
    module="WINAICO WSx-240P6",
    elev=300,
    tracking="fixed",
    inverter=None,
    inverter_kwargs={},
    tracking_args={},
    output_netcdf_path=None,
    output_variables=None,
):
    """

    openfield_pv_era5_unvalidated(placements, era5_path, global_solar_atlas_ghi_path, global_solar_atlas_dni_path, module="WINAICO WSx-240P6", elev=300, tracking="fixed", inverter=None, inverter_kwargs={}, tracking_args={}, output_netcdf_path=None, output_variables=None)


    Simulation of an openfield  PV openfield system based on ERA5 Data.

    Parameters
    ----------
    placements: Pandas Dataframe
                    Locations that you want to do the simulations for.
                    Columns need to be lat (latitudes), lon (longitudes), tilt and capacity.

    era5_path: str
                Path to the ERA5 Data on your computer.
                Can be a single ".nc" file, or a directory containing many ".nc" files.

    global_solar_atlas_ghi_path: str
                                    Path to the global solar atlas ghi data on your computer.

    global_solar_atlas_dni_path: str
                                    Path to the global solar atlas dni data on your computer.

    module: str
            Name of the module that you wanna use for the simulation.
            Default is Winaico Wsx-240P6

    elev: float
            Elevation that you want to model your PV system at.

    tracking: str
                Determines wether your PV system is fixed or not.
                Default is fixed.
                Option 1 is 'fixed' meaning that the module does not have any tracking capabilities.
                Option 2 is 'single-axis' meaning that the module has single-axis tracking capabilities.

    inverter: str
                Determines wether you want to model your PV system with an inverter or not.
                Default is None.
                See reskit.solar.SolarWorkflowManager.apply_inverter_losses for more usage information.

    output_netcdf_path: str
                        Path to a file that you want to save your output NETCDF file at.
                        Default is None

    output_variables: str
                        Output variables of the simulation that you want to save into your NETCDF Outputfile.


    Returns
    -------
    A xarray dataset including all the output variables you defined as your output_variables.

    """

    wf = SolarWorkflowManager(placements)
    wf.configure_cec_module(module)

    if not "tilt" in wf.placements.columns:
        wf.estimate_tilt_from_latitude(convention="Ryberg2020")
    if not "azimuth" in wf.placements.columns:
        wf.estimate_azimuth_from_latitude()
    if not "elev" in wf.placements.columns:
        wf.apply_elevation(elev)

    wf.read(
        variables=[
            "global_horizontal_irradiance",
            "direct_horizontal_irradiance",
            "surface_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
            "surface_dew_temperature",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=False,
    )

    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()

    wf.direct_normal_irradiance_from_trigonometry()

    wf.adjust_variable_to_long_run_average(
        variable="global_horizontal_irradiance",
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=global_solar_atlas_ghi_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )

    wf.adjust_variable_to_long_run_average(
        variable="direct_normal_irradiance",
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_DNI,
        real_long_run_average=global_solar_atlas_dni_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )

    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370)
    wf.determine_air_mass(model="kastenyoung1989")

    wf.diffuse_horizontal_irradiance_from_trigonometry()

    if tracking == "single_axis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sapm()

    wf.simulate_with_interpolated_single_diode_approximation(module=module)

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    wf.apply_loss_factor(0.20, variables=["capacity_factor", "total_system_generation"])

    wf.make_mean(["capacity_factor", "total_system_generation"], fill_na=0)

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def _convert_to_probability_dict(value: Union[float, List[float], Dict[float, float]]) -> Dict[float, float]:
    # Ensure we have a dictionary
    if isinstance(value, float):
        value = {value: 1}
    elif isinstance(value, list):
        value = {v: 1 for v in value}
    elif not isinstance(value, dict):
        raise TypeError("value type must be one of float, List[float], or Dict[float,float]")

    # Normalize the weights in the dictionary
    total_weight = sum(value.values())
    value = {k: v / total_weight for k, v in value.items()}

    # Done!
    return value


def rooftop_pv_era5_unvalidated(
    placements,
    era5_path,
    global_solar_atlas_ghi_path,
    global_solar_atlas_dni_path,
    azimuths: Union[float, List[float], Dict[float, float]] = [90, 135, 180, 225, 270],
    tilts: Union[float, List[float], Dict[float, float]] = {
        28.15: 0.022713075351,
        33.08: 0.135886324455,
        37.70: 0.341329048107,
        42.30: 0.341360442417,
        46.92: 0.135923920404,
        51.857: 0.022723846777,
    },
    module="LG Electronics LG370Q1C-A5",
    elev=300,
    inverter=None,
    inverter_kwargs={},
    output_netcdf_path=None,
    output_variables=None,
):
    """
    """
    # Condition "tilts" and "azimuths" input
    tilts = _convert_to_probability_dict(tilts)
    azimuths = _convert_to_probability_dict(azimuths)

    # Perform orientation-invariant workflow steps
    wf = SolarWorkflowManager(placements)
    wf.configure_cec_module(module)

    if not "elev" in wf.placements.columns:
        wf.apply_elevation(elev)

    wf.read(
        variables=[
            "global_horizontal_irradiance",
            "direct_horizontal_irradiance",
            "surface_wind_speed",
            "surface_pressure",
            "surface_air_temperature",
            "surface_dew_temperature",
        ],
        source_type="ERA5",
        source=era5_path,
        set_time_index=True,
        verbose=False,
    )

    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()

    wf.direct_normal_irradiance_from_trigonometry()

    wf.adjust_variable_to_long_run_average(
        variable="global_horizontal_irradiance",
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=global_solar_atlas_ghi_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )

    wf.adjust_variable_to_long_run_average(
        variable="direct_normal_irradiance",
        source_long_run_average=rk_weather.Era5Source.LONG_RUN_AVERAGE_DNI,
        real_long_run_average=global_solar_atlas_dni_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )

    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370)
    wf.determine_air_mass(model="kastenyoung1989")

    wf.diffuse_horizontal_irradiance_from_trigonometry()

    # Perform orientation-aware workflow steps, and aggregate
    capacity_factor = 0
    total_system_generation = 0

    for azimuth, azimuth_weight in azimuths.items():
        for tilt, tilt_weight in tilts.items():
            # Set the new tilt and azimuth value
            wf.placements["tilt"] = tilt
            wf.placements["azimuth"] = azimuth

            # Perform simulation
            wf.determine_angle_of_incidence()
            wf.estimate_plane_of_array_irradiances(transposition_model="perez")

            wf.apply_angle_of_incidence_losses_to_poa()

            wf.cell_temperature_from_sapm(mounting="glass_close_roof")

            wf.simulate_with_interpolated_single_diode_approximation(module=module)

            if inverter is not None:
                wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

            wf.apply_loss_factor(0.20, variables=["capacity_factor", "total_system_generation"])

            # increment the output capacity_factor and total_system_generation outputs
            capacity_factor += wf.sim_data["capacity_factor"] * azimuth_weight * tilt_weight
            total_system_generation += wf.sim_data["total_system_generation"] * azimuth_weight * tilt_weight

            # Drop the variables created in the above simulation steps
            del wf.sim_data["angle_of_incidence"]
            del wf.sim_data["poa_global"]
            del wf.sim_data["poa_direct"]
            del wf.sim_data["poa_diffuse"]
            del wf.sim_data["poa_sky_diffuse"]
            del wf.sim_data["poa_ground_diffuse"]
            del wf.sim_data["cell_temperature"]
            del wf.sim_data["module_dc_power_at_mpp"]
            del wf.sim_data["module_dc_voltage_at_mpp"]
            del wf.sim_data["capacity_factor"]
            del wf.sim_data["total_system_generation"]

    # Set the final capacity_factor and total_system_generation data
    wf.sim_data["capacity_factor"] = capacity_factor
    wf.sim_data["total_system_generation"] = total_system_generation

    wf.make_mean(["capacity_factor", "total_system_generation"], fill_na=0)

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)


def openfield_pv_sarah_unvalidated(
    placements,
    sarah_path,
    era5_path,
    module="WINAICO WSx-240P6",
    elev=300,
    tracking="fixed",
    inverter=None,
    inverter_kwargs={},
    tracking_args={},
    output_netcdf_path=None,
    output_variables=None,
):
    """

    openfield_pv_sarah_unvalidated(placements, sarah_path, era5_path, module="WINAICO WSx-240P6", elev=300, tracking="fixed", inverter=None, inverter_kwargs={}, tracking_args={}, output_netcdf_path=None, output_variables=None)


    Simulation of an openfield  PV openfield system based on Sarah and ERA5 Data.

    Parameters
    ----------
    placements: Pandas Dataframe
                    Locations that you want to do the simulations for.
                    Columns need to be lat (latitudes), lon (longitudes), tilt and capacity.

    sarah_path: str
                Path to the SARAH Data on your computer.
                Can be a single ".nc" file, or a directory containing many ".nc" files.

    era5_path: str
                Path to the ERA5 Data on your computer.
                Can be a single ".nc" file, or a directory containing many ".nc" files.


    module: str
            Name of the module that you wanna use for the simulation.
            Default is Winaico Wsx-240P6

    elev: float
            Elevation that you want to model your PV system at.

    tracking: str
                Determines wether your PV system is fixed or not.
                Default is fixed.
                Option 1 is 'fixed' meaning that the module does not have any tracking capabilities.
                Option 2 is 'single-axis' meaning that the module has single-axis tracking capabilities.

    inverter: str
                Determines wether you want to model your PV system with an inverter or not.
                Default is None.
                See reskit.solar.SolarWorkflowManager.apply_inverter_losses for more usage information.

    output_netcdf_path: str
                        Path to a file that you want to save your output NETCDF file at.
                        Default is None

    output_variables: str
                        Output variables of the simulation that you want to save into your NETCDF Outputfile.


    Returns
    -------
    A xarray dataset including all the output variables you defined as your output_variables.

    """

    wf = SolarWorkflowManager(placements)
    wf.configure_cec_module(module)

    if not "tilt" in wf.placements.columns:
        wf.estimate_tilt_from_latitude(convention="Ryberg2020")
    if not "azimuth" in wf.placements.columns:
        wf.estimate_azimuth_from_latitude()
    if not "elev" in wf.placements.columns:
        wf.apply_elevation(elev)

    wf.read(
        variables=["direct_normal_irradiance", "global_horizontal_irradiance"],
        source_type="SARAH",
        source=sarah_path,
        set_time_index=True,
        verbose=False,
    )

    wf.read(
        variables=["surface_wind_speed", "surface_pressure", "surface_air_temperature", "surface_dew_temperature",],
        source_type="ERA5",
        source=era5_path,
        set_time_index=False,
        verbose=False,
    )

    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()
    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370)
    wf.determine_air_mass(model="kastenyoung1989")

    wf.diffuse_horizontal_irradiance_from_trigonometry()

    if tracking == "single_axis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sapm()

    wf.simulate_with_interpolated_single_diode_approximation(module=module)

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    wf.apply_loss_factor(0.20, variables=["capacity_factor", "total_system_generation"])

    return wf.to_xarray(output_netcdf_path=output_netcdf_path, output_variables=output_variables)

