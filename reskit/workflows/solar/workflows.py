import reskit as rk
from .solar_workflow_generator import SolarWorkflowGenerator


def openfield_pv_with_merra_ryberg2019_europe(placements, merra_path, global_solar_atlas_ghi_path, module="WINAICO WSx-240P6", elev=300, tracking="fixed", inverter=None, inverter_kwargs={}, tracking_args={}, output_netcdf_path=None):
    wf = SolarWorkflowGenerator(placements)

    if not "elev" in wf.placements.columns:
        wf.apply_elevation(elev)

    wf.read(
        variables=['surface_wind_speed',
                   "surface_pressure",
                   "surface_air_temperature",
                   "surface_dew_temperature",
                   "global_horizontal_irradiance"],
        source_type="MERRA",
        path=merra_path,
        set_time_index=True,
        verbose=False
    )

    wf.adjust_variable_to_long_run_average(
        variable='global_horizontal_irradiance',
        source_long_run_average=rk.weather.sources.MerraSource.LONG_RUN_AVERAGE_GHI,
        real_long_run_average=global_solar_atlas_ghi_path,
        real_lra_scaling=1000 / 24,  # cast to hourly average kWh
    )
    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()
    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370)
    wf.determine_air_mass(model='kastenyoung1989')
    wf.apply_DIRINT_model()
    wf.diffuse_horizontal_irradiance_from_trigonometry()

    if tracking == "singleaxis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sandia_method()

    wf.simulate_with_interpolated_single_diode_approximation()

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    wf.apply_loss_factor(0.20, variables=['capacity_factor', 'total_system_generation'])

    return wf.to_xarray(output_netcdf_path=output_netcdf_path)


def openfield_pv_with_sarah_ryberg2020(placements, sarah_path, era5_path, module="WINAICO WSx-240P6", elev=300, tracking="fixed", inverter=None, inverter_kwargs={}, tracking_args={}, output_netcdf_path=None):
    wf = SolarWorkflowGenerator(placements)

    if not "elev" in wf.placements.columns:
        wf.apply_elevation(elev)

    wf.read(
        variables=["direct_normal_irradiance",
                   "global_horizontal_irradiance"],
        source_type="SARAH",
        path=sarah_path,
        set_time_index=True,
        verbose=False
    )

    wf.read(
        variables=["surface_wind_speed",
                   "surface_pressure",
                   "surface_air_temperature",
                   "surface_dew_temperature", ],
        source_type="ERA5",
        path=era5_path,
        set_time_index=False,
        verbose=False
    )

    wf.determine_solar_position()
    wf.filter_positive_solar_elevation()
    wf.determine_extra_terrestrial_irradiance(model="spencer", solar_constant=1370)
    wf.determine_air_mass(model='kastenyoung1989')

    wf.diffuse_horizontal_irradiance_from_trigonometry()

    if tracking == "singleaxis":
        wf.permit_single_axis_tracking(**tracking_args)

    wf.determine_angle_of_incidence()
    wf.estimate_plane_of_array_irradiances(transposition_model="perez")

    wf.apply_angle_of_incidence_losses_to_poa()

    wf.cell_temperature_from_sandia_method()

    wf.simulate_with_interpolated_single_diode_approximation()

    if inverter is not None:
        wf.apply_inverter_losses(inverter=inverter, **inverter_kwargs)

    wf.apply_loss_factor(0.20, variables=['capacity_factor', 'total_system_generation'])

    return wf.to_xarray(output_netcdf_path=output_netcdf_path)