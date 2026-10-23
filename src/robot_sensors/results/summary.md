# Offline validation summary

## A. Geometry (raycast vs hand-computed)
max |error| = 0.00e+00 m

## C. Per-beam Monte Carlo vs model
- alpha_sweep_max_dropout_err: 0.0013
- alpha_sweep_max_sigma_rel_err: 0.0069
- range_sweep_max_dropout_err: 0.0024
- range_sweep_max_sigma_rel_err: 0.0070

## D. Fog mixture
max |empirical − predicted| mass = 0.0156

## E. Observability (rank of Fisher information, state = (x, y, psi))
- wall: rank 2, eigenvalues ['0', '9.25e+05', '1.15e+07'], weakest direction ['0.000', '-1.000', '0.000']
- corner: rank 3, eigenvalues ['6.97e+05', '1.24e+06', '1.31e+07'], weakest direction ['0.996', '-0.067', '-0.062']

## F. Estimator Monte Carlo vs inverse Fisher information (corner scene)
| beta | sd_x pred / emp [mm] | sd_y pred / emp [mm] | sd_psi pred / emp [mdeg] | ratios (emp/pred) |
|---|---|---|---|---|
| 0 | 1.19 / 1.27 | 0.90 / 0.94 | 16.53 / 17.43 | 1.07, 1.05, 1.05 |
| 0.03 | 1.35 / 1.28 | 0.98 / 0.97 | 19.37 / 19.36 | 0.95, 0.98, 1.00 |
| 0.06 | 1.55 / 1.51 | 1.08 / 1.01 | 23.27 / 24.12 | 0.98, 0.93, 1.04 |
| 0.1 | 1.86 / 1.87 | 1.24 / 1.24 | 29.77 / 29.42 | 1.00, 1.00, 0.99 |

## G. Optimal incidence for distance information
- D=1: {'gamma_opt_deg': 61.50000000000001, 'gain_vs_normal': 2.0976591807734875}
- D=2: {'gamma_opt_deg': 45.0, 'gain_vs_normal': 1.2025141206381396}
- D=4: {'gamma_opt_deg': 0.0, 'gain_vs_normal': 1.0}

## H. CRB map (clutter)
- beta=0: {'median_mm': 1.1038433759547086, 'max_mm': 1.4368043850522667}
- beta=0.08: {'median_mm': 1.376611869971111, 'max_mm': 2.1718471694688457}

## I. FOV / LOS / detection coverage (270 deg, rho = 0.1)
- poseA: {'pose': [-3.5, -3.5, 0.7853981633974483], 'area_free_m2': 97.36000000000001, 'area_fov_m2': 95.08000000000001, 'area_fov_los_m2': 68.47000000000001, 'area_obs_beta=0_m2': 40.50000000000001, 'area_obs_beta=0.1_m2': 21.097500000000004}
- poseB: {'pose': [0.0, -1.0, 1.5707963267948966], 'area_free_m2': 97.36000000000001, 'area_fov_m2': 81.33250000000001, 'area_fov_los_m2': 62.09750000000001, 'area_obs_beta=0_m2': 58.860000000000014, 'area_obs_beta=0.1_m2': 31.215000000000007}
- r_half_m: {'beta=0': 6.2470587327689024, 'beta=0.1': 3.9879871452279625}