# Batch processing for very preliminary TTV search
# Python 3.8.2
# TODO:
# Assign each observation its own instrument parameters and make this work with fitting
# Need to add in radial velocity considerations, juliet docs have more info for RV + Transit fits
import argparse
import os
from os.path import split
import pickle
from pathlib import Path

import astropy.units as u
import juliet
import lightkurve as lk
from matplotlib import font_manager
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares, LombScargle
from brokenaxes import brokenaxes
from ldtk import LDPSetCreator
from ldtk.filters import tess
from scipy import stats


# Whether or not to do the TTV fit
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TTV and/or GP analysis for TESS targets."
    )
    # TTV flag
    parser.add_argument(
        "--no_ttv",
        action="store_false",  # Sets do_ttv=False when flag is present
        dest="do_ttv",
        default=True,
        help="Disable TTV fitting (default: perform TTV fit)",
    )

    # GP Kernel Choice
    # Mutally exclusive group to choose only one GP kernel, or neither
    gp_group = parser.add_mutually_exclusive_group()

    gp_group.add_argument(
        "--gp_qp",
        action="store_true",
        dest="gp_qp",
        default=False,
        help="Enable GP fit with the quasi-periodic kernel",
    )

    gp_group.add_argument(
        "--gp_matern",
        action="store_true",
        dest="gp_matern",
        default=False,
        help="Enable GP fit with the approximate Matern kernel",
    )

    return parser.parse_args()


args = parse_args()


# Set n_live_points based on free parameters
def calc_dimensions(priors, key="distribution", distribution="fixed"):
    dimensions = 0
    if isinstance(priors, dict):
        for k, v in priors.items():
            if k == key and v != distribution:
                dimensions += 1
            dimensions += calc_dimensions(v)
    return dimensions


def calc_n_live_points(dimensions):
    return int(np.ceil((dimensions**2) / 100)) * 100


# Read Full CSV
all_data = pd.read_csv("systems.csv", comment="#")
system_names = all_data["system_name"].dropna().unique().tolist()  # Get unique names

# Likelihood Results File Setup
# Define the path for the likelihood results CSV
like_csv_file = "likelihood_results.csv"
# Define expected columns for the results DataFrame
expected_like_cols = [
    "system_name",
    "logZ_ttv",
    "logZ_ttv_err",
    "logZ_no_ttv",
    "logZ_no_ttv_err",
    "delta_logZ",
]

# Try to load existing file, or create an empty DataFrame with correct columns
try:
    df_likelihoods = pd.read_csv(like_csv_file)
    # Check and add missing columns if the file exists but is incomplete
    for col in expected_like_cols:
        if col not in df_likelihoods.columns:
            df_likelihoods[col] = np.nan
except FileNotFoundError:
    df_likelihoods = pd.DataFrame(columns=expected_like_cols)

# Start of Analysis
for system_name in system_names:
    print(f"Processing {system_name}...")
    # Get the row corresponding to the current system_name
    data = all_data[all_data["system_name"] == system_name].iloc[0]

    # Extract parameters for this system
    P = float(data["period"])
    a = float(data["a"])
    p = float(data["p"])
    b = float(data["b"])
    teff = float(data["teff"])
    logg = float(data["logg"])
    feh = float(data["feh"])

    # Create folder for system results
    result_path = f"{system_name}_results"
    result_dir = Path(result_path)
    result_dir.mkdir(exist_ok=True)

    # Calculate LDCs (needs to be inside loop as it depends on stellar params)
    print(f"Calculating LDCs for {system_name}...")
    sc = LDPSetCreator(
        teff=(teff, 100), logg=(logg, 0.2), z=(feh, 0.05), filters=[tess]
    )
    ps = sc.create_profiles(nsamples=2000)
    ps.resample_linear_z(300)

    qc, qe = ps.coeffs_qd(do_mc=True, n_mc_samples=10000)
    q1, q2 = qc[0][0], qc[0][1]
    # q1err, q2err = qe[0][0], qe[0][1]  # Errors not used in fixed priors

    print(f"Getting lightcurves for {system_name}...")
    # Fetch and prepare data
    search_result = lk.search_lightcurve(system_name, mission="TESS", author="SPOC")

    # Download lightcurves, doing donwload all adds a lot data so longer fits
    # lc_collection_raw = [search_result.download()[:5] # could do something like this if you only want the first five
    lc_collection_raw = search_result.download_all()

    instrument_names = []  # To store names like TESS-SEC01
    times = {}
    fluxes = {}
    fluxes_error = {}
    t_list, f_list, ferr_list = [], [], []

    for lc_individual in lc_collection_raw:
        # It MAY be necessary to save each sector as an
        # 'instrumnet' and have its own parameters but for now I can't get it to
        # work this way, so it is all one instrument. Binning might fix this.
        sector = lc_individual.meta.get("SECTOR", "UNKNOWN")
        instrument_name = f"TESS{sector}"  # Create instrument name
        instrument_names.append(instrument_name)

        lc_processed = (
            lc_individual.normalize()
            .remove_nans()
            .remove_outliers(sigma=5)
            # .bin(5 / 1440)
        )
        # binning might help reduce run time, but not recommended

        if lc_processed is None or len(lc_processed.time) == 0:
            continue

        # Extract and clean data, convert to numpy explicitly
        time_val = lc_processed.time.value
        flux_val = lc_processed.flux.value
        flux_err_val = (
            lc_processed.flux_err.value
            if hasattr(lc_processed, "flux_err") and lc_processed.flux_err is not None
            else np.full_like(time_val, np.nan)
        )

        t_inst = np.asarray(time_val)
        f_inst = np.asarray(flux_val)
        ferr_inst = np.asarray(flux_err_val)

        min_len = min(len(t_inst), len(f_inst), len(ferr_inst))
        t_inst, f_inst, ferr_inst = (
            t_inst[:min_len],
            f_inst[:min_len],
            ferr_inst[:min_len],
        )
        finite_mask = np.isfinite(t_inst) & np.isfinite(f_inst)
        t_inst, f_inst, ferr_inst = (
            t_inst[finite_mask],
            f_inst[finite_mask],
            ferr_inst[finite_mask],
        )
        if len(t_inst) < 10:
            continue  # Skip if too few points remain

        # Store time and flux in dictionary for juliet
        times.update({instrument_name: t_inst})
        fluxes.update({instrument_name: f_inst})
        fluxes_error.update({instrument_name: ferr_inst})

        # Store in list for now
        t_list.extend(t_inst)
        f_list.extend(f_inst)
        ferr_list.extend(ferr_inst)

        # Prepare data for concatenation (handle errors for BLS)
        # median_err = np.nanmedian(
        #     ferr_inst[np.isfinite(ferr_inst) & (ferr_inst > 0)]
        # )  # Calculate median from valid errors, not sure I need this

        # if np.isnan(median_err) or median_err <= 0:
        #     median_err = 1.0  # Absolute fallback
        # ferr_bls = np.copy(ferr_inst)  # Make a copy for BLS
        # invalid_err_mask_bls = ~np.isfinite(ferr_bls) | (ferr_bls <= 0)
        # ferr_bls[invalid_err_mask_bls] = (
        #     median_err  # Replace invalid with median for BLS
        # )
        # all_ferr_for_estimate.append(ferr_bls)

    # Create and sort arrays
    t = np.array(t_list)
    f = np.array(f_list)
    ferr = np.array(ferr_list)

    # Have to sort for GP, juliet says it does this internally, but it lies
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]
    f_sorted = f[sort_idx]
    ferr_sorted = ferr[sort_idx]  # Original errors, sorted

    # Detect gaps and create segments
    gap_threshold = 7.0  # gaps >7 days will be removed AFFECTS ONLY THE PLOTTING
    buffer = 0.5
    dt = np.diff(t)
    gap_indices = np.where(dt > gap_threshold)[0]
    split_indices = gap_indices + 1
    t_segments = np.split(t, split_indices)
    f_segments = np.split(f, split_indices)
    ferr_segments = np.split(ferr, split_indices)

    xlims = []
    for t_seg in t_segments:
        if len(t_seg) > 0:
            xlims.append((t_seg[0] - buffer, t_seg[-1] + buffer))

    # Create broken axes plot
    fig = plt.figure(figsize=(10, 5))
    bax = brokenaxes(xlims=xlims, hspace=0.1, despine=False)

    # Plot each segment with errorbars
    for t_seg, f_seg, ferr_seg in zip(t_segments, f_segments, ferr_segments):
        if len(t_seg) > 0:
            bax.errorbar(
                t_seg,
                f_seg,
                yerr=ferr_seg,
                fmt=".",
                color="royalblue",
                markersize=2,
                alpha=0.5,
            )

    # Add labels and styling to brokenaxes object
    bax.set_xlabel(f"Time ({lc_processed.time.format.upper()})", fontsize=12)
    bax.set_ylabel("Normalized Flux", fontsize=12)
    bax.set_title(f"TESS Light Curve: {system_name}", fontsize=14)
    bax.grid(alpha=0.3)

    # Save and close
    plt.savefig(
        f"{result_path}/lightcurve_{system_name}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Estimate t0
    print(f"Calculating t0 for {system_name}")
    period_search_min = P * 0.99  # Using P from CSV
    period_search_max = P * 1.01

    P_for_duration_est = P
    central_duration_est = 0.1 * (P_for_duration_est ** (1.0 / 3.0))
    duration_min_auto = max(0.01, central_duration_est * 0.3)
    duration_max_auto = central_duration_est * 1.75
    duration_max_auto = min(duration_max_auto, period_search_min / 2.0 * 0.95)
    if duration_min_auto >= duration_max_auto:
        duration_min_auto = max(0.01, duration_max_auto * 0.5)

    bls_periods = np.linspace(period_search_min, period_search_max, 10000)
    bls_durations = np.linspace(duration_min_auto, duration_max_auto, 20)

    # Run BLS (with basic fallback)
    bls_model = BoxLeastSquares(t * u.day, f, ferr)
    P_bls = P  # Default period
    transit_duration_bls = central_duration_est  # Default duration
    try:
        bls_results = bls_model.power(bls_periods * u.day, bls_durations * u.day)
        if bls_results is None or len(bls_results.period) == 0:
            raise ValueError("BLS empty")
        index = np.argmax(bls_results.power)
        P_bls = bls_results.period[index].value
        transit_duration_bls = bls_results.duration[index].value
    except Exception:  # Catch any BLS error
        pass  # Silently use default P and duration if BLS fails

    # Phase Folding
    t0_bls = bls_results.transit_time[index].value
    phase = (t - t0_bls) / P_bls % 1.0
    phase = np.where(phase >= 0.5, phase - 1.0, phase)

    n_bins = 200
    phase_min_bin = 0.0  # Default
    bin_centers = None  # Initialize
    bin_means = None  # Initialize
    # Check for enough finite points for binning
    valid_mask_binning = np.isfinite(phase) & np.isfinite(f)
    if np.sum(valid_mask_binning) >= n_bins:
        try:
            bin_means, bin_edges, _ = stats.binned_statistic(
                phase[valid_mask_binning],
                f[valid_mask_binning],
                statistic="mean",
                bins=n_bins,
            )
            if not np.all(np.isfinite(bin_means)):
                min_bin_index = np.nanargmin(bin_means)
            else:
                min_bin_index = np.argmin(bin_means)
            bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0
            phase_min_bin = bin_centers[min_bin_index]
        except Exception:  # Catch binning errors
            phase_min_bin = 0.0  # Use default if binning fails

    t0_estimate = t0_bls + phase_min_bin * P_bls  # Preliminary t0

    # Adjusting t0 to nearest transit with data
    search_cycles = 2
    search_width = transit_duration_bls * 1.5
    min_points_in_window = 30
    predicted_times = [
        t0_estimate + m * P_bls for m in range(-search_cycles, search_cycles + 1)
    ]
    best_observed_t0 = None
    min_distance_to_t0_bls = np.inf
    for i, predicted_t in enumerate(predicted_times):
        t_start_window = predicted_t - search_width / 2.0
        t_end_window = predicted_t + search_width / 2.0
        mask = (t >= t_start_window) & (t <= t_end_window)
        n_points = np.sum(mask)
        if n_points >= min_points_in_window:
            times_in_window = t[mask]
            fluxes_in_window = f[mask]
            finite_flux_mask = np.isfinite(fluxes_in_window)
            if np.sum(finite_flux_mask) > 0:
                try:
                    min_idx_in_finite = np.argmin(fluxes_in_window[finite_flux_mask])
                    t_min_local = times_in_window[finite_flux_mask][min_idx_in_finite]
                    distance = abs(t_min_local - t0_bls)
                    if distance < min_distance_to_t0_bls:
                        min_distance_to_t0_bls = distance
                        best_observed_t0 = t_min_local
                except IndexError:
                    pass  # Should be prevented

    # Select Final t0 Estimate
    if best_observed_t0 is not None:
        t0_estimate_final = best_observed_t0
    else:
        t0_estimate_final = (
            t0_estimate  # Use the preliminary estimate if snapping failed
        )

    # Plot t0 Check
    plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = plt.subplot(gs[0])
    ax1.plot(
        phase, f, ".", markersize=2, color="royalblue", alpha=0.3, label="All Data"
    )

    if bin_centers is not None and bin_means is not None:  # Check if binning happened
        ax1.plot(bin_centers, bin_means, "o-", color="red", label="Binned Data")
        ax1.axvline(
            phase_min_bin,
            color="k",
            linestyle="--",
            label=f"Min Phase ({phase_min_bin:.3f})",
        )

    ax1.set_xlabel(f"Phase (P={P_bls:.6f} days)")
    ax1.set_ylabel("Flux")
    ax1.set_title("Phase-Folded Light Curve")
    ax1.legend()
    ax1.grid(True)

    # Plot original data segments
    ax2 = brokenaxes(
        xlims=xlims, hspace=0.1, despine=False, subplot_spec=gs[1], diag_color="none"
    )

    for t_seg, f_seg in zip(t_segments, f_segments):
        if len(t_seg) > 0:
            ax2.plot(
                t_seg,
                f_seg,
                ".",
                color="royalblue",
                markersize=2,
                alpha=0.5,
            )

    # Add vertical lines to broken axes
    ax2.axvline(
        t0_bls, color="grey", linestyle=":", lw=2, label=f"t0_bls ({t0_bls:.3f})"
    )
    ax2.axvline(
        t0_estimate,
        color="orange",
        linestyle="--",
        lw=1.5,
        label=f"t0_prelim ({t0_estimate:.3f})",
    )
    ax2.axvline(
        t0_estimate_final,
        color="red",
        linestyle="-",
        lw=2,
        label=f"t0_final ({t0_estimate_final:.3f})",
    )

    # Format broken axes
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Flux")
    ax2.set_title("Original Light Curve with t0 Estimates")
    ax2.legend()
    ax2.grid(True)

    # Finalize
    plt.tight_layout()
    plt.savefig(
        f"{result_path}/check_t0_for_{system_name}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Populate priors for GP if selected
    if args.gp_qp or args.gp_matern:
        print(f"GP fit for {system_name}...")
        # Following juliet docs, need to mask out in transit data, have to do this dynamically for each system
        # Ensures that ingress and egress points are caputred, ~ 1.5 - 2 are good values
        mask_buffer_factor = 2

        phase_duration_to_mask = (transit_duration_bls * mask_buffer_factor) / P_bls
        phase_threshold = phase_duration_to_mask / 2

        # Never mask more than +/- 0.1 phase
        phase_threshold = min(phase_threshold, 0.1)

        # Dictionaries for fitting
        times_oot, fluxes_oot, fluxes_error_oot = {}, {}, {}

        # List for plotting
        phases_it_all, f_it_all = [], []
        phases_oot_all, f_oot_all = [], []

        # Now iterate over each instrument dataset
        for instrument in instrument_names:
            t_inst = times[instrument]
            f_inst = fluxes[instrument]
            ferr_inst = fluxes_error[instrument]

            phases_inst = juliet.utils.get_phases(t_inst, P_bls, t0_estimate_final)

            idx_oot_inst = np.where(np.abs(phases_inst) > phase_threshold)[0]
            idx_it_inst = np.where(np.abs(phases_inst) <= phase_threshold)[
                0
            ]  # Indices for in-transit points (for plotting check)

            phases_oot_all.extend(phases_inst[idx_oot_inst])
            f_oot_all.extend(f_inst[idx_oot_inst])
            phases_it_all.extend(phases_inst[idx_it_inst])
            f_it_all.extend(f_inst[idx_it_inst])

            times_oot[instrument] = t_inst[idx_oot_inst]
            fluxes_oot[instrument] = f_inst[idx_oot_inst]
            fluxes_error_oot[instrument] = ferr_inst[idx_oot_inst]

        phases_oot_all = np.array(phases_oot_all)
        f_oot_all = np.array(f_oot_all)
        phases_it_all = np.array(phases_it_all)
        f_it_all = np.array(f_it_all)

        plt.figure(figsize=(10, 6))
        plt.plot(
            phases_oot_all,
            f_oot_all,
            "k.",
            alpha=0.1,
            label="OOT Data (used for GP)",
        )
        plt.plot(
            phases_it_all,
            f_it_all,
            "r.",
            alpha=0.3,
            label="In-Transit Data (masked)",
        )
        plt.xlabel(f"Phase (P={P_bls:.6f} d)")
        plt.ylabel("Normalized Flux")
        plt.title(
            f"{system_name} - Transit Masking Check (Threshold: +/- {phase_threshold:.4f})"
        )
        plt.legend()
        plt.grid(True)
        plt.xlim(-0.15, 0.15)  # Zoom in near transit
        plt.savefig(
            f"{result_path}/check_oot_mask_{system_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        priors = {}
        for instrument in instrument_names:
            priors[f"mdilution_{instrument}"] = {
                "distribution": "fixed",
                "hyperparameters": 1.0,
            }  # Keep fixed=1 as original
            priors[f"mflux_{instrument}"] = {
                "distribution": "normal",
                "hyperparameters": [0.0, 0.1],
            }
            priors[f"sigma_w_{instrument}"] = {
                "distribution": "loguniform",
                "hyperparameters": [1e-5, 0.1],
            }
            # Priors for GPs
            # These are the Matern kernel params
            if args.gp_matern:
                variance_est = np.var(fluxes_oot[instrument])
                amp_low = max(1e-8, variance_est / 100.0)
                amp_high = min(1e-2, variance_est * 100.0)
                priors[f"GP_sigma_{instrument}"] = {
                    "distribution": "loguniform",
                    "hyperparameters": [amp_low, amp_high],
                }
                priors[f"GP_rho_{instrument}"] = {
                    "distribution": "loguniform",
                    "hyperparameters": [0.01, 100.0],
                }

            # Quasi periodic GP paramrs, use if there is stellar variability
            if args.gp_qp:
                primary_instrument = instrument_names[
                    0
                ]  # used to share parameters between instruments
                variance_est = np.var(fluxes_oot[instrument])
                amp_low = max(1e-8, variance_est / 100.0)
                amp_high = min(1e-2, variance_est * 100.0)
                priors[f"GP_B_{instrument}"] = {
                    "distribution": "loguniform",
                    "hyperparameters": [amp_low, amp_high],
                }
                priors[f"GP_C_{instrument}"] = {
                    "distribution": "loguniform",
                    "hyperparameters": [1.0, 1000.0],
                }
                priors[f"GP_L_{instrument}"] = {
                    "distribution": "loguniform",
                    "hyperparameters": [1.0, 200.0],
                }
                priors[f"GP_Prot_{instrument}"] = {
                    "distribution": "normal",  # need to add way to estimate Prot
                    "hyperparameters": [5.0, 2.0],
                }

        if args.gp_matern:
            # Perform GP fit with Matern kernel
            dataset = juliet.load(
                priors=priors,
                t_lc=times_oot,
                y_lc=fluxes_oot,
                yerr_lc=fluxes_error_oot,
                GP_regressors_lc=times_oot,
                out_folder=f"{system_name}_results/{system_name}_GP_Matern",
            )

        if args.gp_qp:
            # Perform GP fit with qp kernel
            dataset = juliet.load(
                priors=priors,
                t_lc=times_oot,
                y_lc=fluxes_oot,
                yerr_lc=fluxes_error_oot,
                GP_regressors_lc=times_oot,
                out_folder=f"{system_name}_results/{system_name}_GP_QP",
            )
        points = 50 * calc_dimensions(priors)
        results = dataset.fit(sampler="dynesty", nthreads=4, n_live_points=points)

        all_times = []
        all_fluxes = []
        all_errors = []
        all_models = []

        for instrument in instrument_names:
            # Extract model and data
            model = results.lc.evaluate(instrument)

            # Put everything into list
            all_times.extend(dataset.times_lc[instrument])
            all_fluxes.extend(dataset.data_lc[instrument])
            all_errors.extend(dataset.errors_lc[instrument])
            all_models.extend(model)

        # Plot the median model
        dt = np.diff(all_times)
        gap_indices = np.where(dt > gap_threshold)[0]
        split_indices = gap_indices + 1

        # Split all relevant arrays
        time_segments = np.split(all_times, split_indices)
        data_segments = np.split(all_fluxes, split_indices)
        error_segments = np.split(all_errors, split_indices)
        model_segments = np.split(all_models, split_indices)

        # Create xlims with buffer
        xlims = [
            (seg[0] - buffer, seg[-1] + buffer) for seg in time_segments if len(seg) > 0
        ]

        # Plot GP model and detrended lightcurve
        fig = plt.figure(figsize=(10, 4))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

        # Top subplot: Broken axes time series
        bax = brokenaxes(
            xlims=xlims,
            hspace=0.1,
            despine=False,
            subplot_spec=gs[0],
        )

        # Plot residuals
        bax2 = brokenaxes(
            xlims=xlims,
            hspace=0.1,
            despine=False,
            subplot_spec=gs[1],
        )

        # Plot each segment with errorbars and model
        for t_seg, d_seg, e_seg, m_seg in zip(
            time_segments, data_segments, error_segments, model_segments
        ):
            if len(t_seg) > 0:
                bax.errorbar(
                    t_seg, d_seg, yerr=e_seg, fmt=".", color="dodgerblue", alpha=0.1
                )
                bax.plot(t_seg, m_seg, color="black", zorder=10)

                bax2.errorbar(
                    t_seg,
                    (d_seg - m_seg) * 1e6,
                    yerr=e_seg * 1e6,
                    fmt=".",
                    color="dodgerblue",
                    alpha=0.1,
                )

        # Format broken axes
        bax.set_ylabel("Relative flux")
        bax.set_ylim([0.98, 1.02])

        # Format residuals
        bax2.set_ylabel("Residuals (ppm)", fontsize=6)
        bax2.set_xlabel("Time (BJD - 2457000)")

        fig.savefig(
            f"{result_path}/gp_residuals_{system_name}.png",
            dpi=300,
            bbox_inches="tight",
        )

    exit()
    # Populate priors
    print(f"No TTV fit for {system_name}...")
    priors = {
        "P_p1": {"distribution": "normal", "hyperparameters": [P_bls, 0.1]},
        "t0_p1": {
            "distribution": "normal",
            "hyperparameters": [t0_estimate_final, 0.1],
        },
        "p_p1": {"distribution": "truncatedNormal", "hyperparameters": [p, 0.1, 0, 1]},
        "b_p1": {"distribution": "truncatedNormal", "hyperparameters": [b, 0.1, 0, 1]},
        "a_p1": {"distribution": "normal", "hyperparameters": [a, 1.0]},
        "q1_TESS": {"distribution": "fixed", "hyperparameters": q1},
        "q2_TESS": {"distribution": "fixed", "hyperparameters": q2},
        "sesinomega_p1": {"distribution": "uniform", "hyperparameters": [-1, 1]},
        "secosomega_p1": {"distribution": "uniform", "hyperparameters": [-1, 1]},
    }

    # could replace some params with parameterizations, see juliet docs
    # Instrument-specific parameters
    for instrument in instrument_names:
        priors[f"mdilution_{instrument}"] = {
            "distribution": "fixed",
            "hyperparameters": 1.0,
        }  # Keep fixed=1 as original
        priors[f"mflux_{instrument}"] = {
            "distribution": "normal",
            "hyperparameters": [0.0, 0.1],
        }
        priors[f"sigma_w_{instrument}"] = {
            "distribution": "loguniform",
            "hyperparameters": [0.1, 1000.0],
        }

    # Initial Fit (No TTV)
    out_folder_no_ttv = f"{result_path}/{system_name}_noTTV"
    dataset = juliet.load(
        priors=priors,
        t_lc=times,
        y_lc=fluxes,
        yerr_lc=fluxes_error,
        out_folder=out_folder_no_ttv,
    )

    # Dynamically assinging n_live_points (wow)
    live_points = 50 * calc_dimensions(priors)
    # I don't know which one is faster, I can't get mulitthreading to work with multinest so maybe
    # dynesty is better

    # results_no_ttv = dataset_no_ttv.fit(
    #    sampler="multinest", use_MPI=True, n_live_points=1000
    # )
    print(f"{calc_dimensions(priors)} free parameters")
    results = dataset.fit(sampler="dynesty", nthreads=4, n_live_points=live_points)

    # Extract Likelihood - with dynesty
    raw_results_file_no_ttv = os.path.join(
        out_folder_no_ttv, "_dynesty_NS_posteriors.pkl"
    )

    dynesty_results_no_ttv = pickle.load(open(raw_results_file_no_ttv, "rb"))

    like_no_ttv, like_no_ttv_err = (
        dynesty_results_no_ttv["lnZ"],
        dynesty_results_no_ttv["lnZerr"],
    )

    # Extract Likelihood (TTV) - with mulitnest
    # like_no_ttv, like_no_ttv_err = (
    #    results_no_ttv.posteriors["lnZ"],
    #    results_no_ttv.posteriors["lnZerr"],
    # )

    # To plot the phased lighcurve we need the median period and time-of-transit center:
    P, t0 = (
        np.median(results.posteriors["posterior_samples"]["P_p1"]),
        np.median(results.posteriors["posterior_samples"]["t0_p1"]),
    )
    all_times = []
    all_fluxes = []
    all_errors = []
    all_models = []
    all_phases = []

    for instrument in instrument_names:
        # Extract full model (transit + GP)
        model = results.lc.evaluate(instrument)

        # Get phases:
        phases = juliet.get_phases(dataset.times_lc[instrument], P, t0)

        # Put everything into list
        all_times.extend(dataset.times_lc[instrument])
        all_fluxes.extend(dataset.data_lc[instrument])
        all_errors.extend(dataset.errors_lc[instrument])
        all_models.extend(model)
        all_phases.extend(phases)

    # Turn all list into arrays
    all_times = np.asarray(all_times)
    all_fluxes = np.asarray(all_fluxes)
    all_errors = np.asarray(all_errors)
    all_models = np.asarray(all_models)
    all_phases = np.asarray(all_phases)

    # Plot the data. First, time versus flux --- plot only the median model here:
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax1 = plt.subplot(gs[0])

    # Plot the median model
    dt = np.diff(all_times)
    gap_indices = np.where(dt > gap_threshold)[0]
    split_indices = gap_indices + 1

    # Split all relevant arrays
    time_segments = np.split(all_times, split_indices)
    data_segments = np.split(all_fluxes, split_indices)
    error_segments = np.split(all_errors, split_indices)
    model_segments = np.split(all_models, split_indices)

    # Create xlims with buffer
    xlims = [
        (seg[0] - buffer, seg[-1] + buffer) for seg in time_segments if len(seg) > 0
    ]

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # Left subplot: Broken axes time series
    bax = brokenaxes(
        xlims=xlims,
        hspace=0.1,
        despine=False,
        subplot_spec=gs[0],
    )

    # Plot each segment with errorbars and model
    for t_seg, d_seg, e_seg, m_seg in zip(
        time_segments, data_segments, error_segments, model_segments
    ):
        if len(t_seg) > 0:
            bax.errorbar(
                t_seg, d_seg, yerr=e_seg, fmt=".", color="dodgerblue", alpha=0.1
            )
            bax.plot(t_seg, m_seg, color="black", zorder=10)

    # Format broken axes
    bax.set_xlabel("Time (BJD - 2457000)")
    bax.set_ylabel("Relative flux")
    bax.set_ylim([0.95, 1.05])

    # Plotting model
    # Now plot phased model; plot the error band of the best-fit model here:
    ax2 = plt.subplot(gs[1])
    ax2.errorbar(
        all_phases,
        all_fluxes,
        yerr=all_errors,
        fmt=".",
        alpha=0.3,
    )
    idx = np.argsort(all_phases)
    ax2.plot(all_phases[idx], all_models[idx], color="black", zorder=10)
    ax2.set_xlabel("Phases")
    ax2.set_xlim([-0.5, 0.5])
    ax2.set_ylim([0.98, 1.02])

    ax1.set_title(f"{system_name} - No TTV Fit")

    fig.savefig(
        f"{result_path}/transit_lightcurve_{system_name}_noTTV.png",
        dpi=300,
        bbox_inches="tight",
    )

    if not args.do_ttv:
        continue

    print(f"TTV fit for {system_name}...")
    # Checking for transit at expected times
    valid_transits = []
    time_min, time_max = np.min(t), np.max(t)

    # Add small buffer to avoid precision issues with floor/ceil
    buffer = 1e-6
    n_min = int(np.floor((time_min - t0) / P + buffer))
    n_max = int(np.ceil((time_max - t0) / P - buffer))
    transit_numbers = np.arange(n_min, n_max + 1)
    expected_times = t0 + transit_numbers * P

    window = transit_duration_bls * 1.2  # Using BLS duration for window
    min_points_for_ttv = 5
    for n, t_pred in zip(transit_numbers, expected_times):
        instrument_found = None
        # Check EACH instrument stored in the 'times' dictionary
        for instrument in instrument_names:  # Use the list of processed instruments
            t_inst = times[instrument]
            mask = (t_inst >= t_pred - window / 2.0) & (t_inst <= t_pred + window / 2.0)
            if np.sum(mask) > min_points_for_ttv:
                instrument_found = instrument
                break  # Found data in this instrument, use it

        if instrument_found:
            valid_transits.append(
                {
                    "transit_number": n,
                    "expected_time": t_pred,
                    "instrument": instrument_found,  # Store the correct instrument name
                }
            )

    # Proceed with TTV fit only if valid transits were found
    if valid_transits:
        print(f"Found {len(valid_transits)} transits for {system_name}...")
        # Set priors for TTV fit
        transit_priors = {}
        for vt in valid_transits:
            param = f"T_p1_{vt['instrument']}_{vt['transit_number']}"
            hypr = {
                "distribution": "normal",
                "hyperparameters": [vt["expected_time"], 0.1],
            }
            transit_priors[param] = hypr

        priors.update(transit_priors)  # Update priors dictionary once

        # Plot Expected Transits
        valid_times = []
        valid_numbers = []
        for vt in valid_transits:
            valid_times.append(vt["expected_time"])
            valid_numbers.append(vt["transit_number"])

        plt.figure(figsize=(12, 6))
        bax = brokenaxes(xlims=xlims, hspace=0.1, despine=False, diag_color="none")

        # Plot light curve with existing data
        bax.errorbar(t, f, fmt=".", alpha=0.3, label="Light Curve")

        window_plot = 0.1
        min_flux_plot = np.percentile(f, 1) if len(f) > 0 else 0.95

        # Plot transit markers and lines
        for t_pred in valid_times:
            mask_plot = (t >= t_pred - window_plot) & (t <= t_pred + window_plot)
            bax.plot(t[mask_plot], f[mask_plot], "ro", alpha=0.7, markersize=5)

            bax.axvline(t_pred, color="k", ls="--", alpha=0.5)

        for n, t_pred in zip(valid_numbers, valid_times):
            for ax in bax.axs:
                xmin, xmax = ax.get_xlim()
                if xmin <= t_pred <= xmax:
                    ax.text(
                        t_pred,
                        min_flux_plot - 0.01,
                        f"N={transit_numbers[n]}",
                        ha="center",
                        va="top",
                        rotation=90,
                        fontsize=4,
                    )
                    break

        # Formatting
        bax.set_xlabel(f"Time ({lc_processed.time.format.upper()})")
        bax.set_ylabel("Normalized Flux")
        bax.set_title(f"Light Curve with Expected Transits: {system_name}")
        bax.legend(loc="upper right")

        plt.savefig(
            f"{result_path}/check_transits_{system_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Fit with TTVs
        out_folder_ttv = f"{result_path}/{system_name}_ttv"  # Unique folder
        dataset = juliet.load(
            priors=priors,
            t_lc=times,
            y_lc=fluxes,
            yerr_lc=fluxes_error,
            GP_regressors_lc=times,
            out_folder=out_folder_ttv,
        )

        live_points_ttv = 50 * calc_dimensions(priors)
        results = dataset.fit(
            sampler="dynesty", nthreads=2, n_live_points=live_points_ttv
        )
        # results_ttv = dataset_ttv.fit(
        #    sampler="multinest", use_MPI=True, n_live_points=100000
        # )

        # Extract Likelihood (TTV)
        raw_results_file_ttv = os.path.join(
            out_folder_ttv, "_dynesty_NS_posteriors.pkl"
        )

        dynesty_results_ttv = pickle.load(open(raw_results_file_ttv, "rb"))

        like_ttv, like_ttv_err = (
            dynesty_results_ttv["lnZ"],
            dynesty_results_ttv["lnZerr"],
        )

        # like_ttv, like_ttv_err = (
        #    results_ttv.posteriors["lnZ"],
        #    results_ttv.posteriors["lnZerr"],
        # )

        # Plot TTV fit results
        # Full model
        transit_plus_GP_ttv = results.lc.evaluate("TESS")

        # Transit model
        # shouldn't need to extract these if not plotting phases
        # transit_model_ttv = results_ttv.lc.model["TESS"]["deterministic"]

        # GP model
        # gp_model_ttv = results_ttv.lc.model["GP"]

        fig = plt.figure(figsize=(12, 4))

        # Plot the model
        time = dataset.times_lc["TESS"]
        dt = np.diff(time)

        time_segments = np.split(time, split_indices)
        data_segments = np.split(dataset.data_lc["TESS"], split_indices)
        error_segments = np.split(dataset.errors_lc["TESS"], split_indices)
        model_segments = np.split(transit_plus_GP_ttv, split_indices)
        # transit_model_segments = np.split(transit_model_ttv, split_indices)
        # gp_model_segments = np.split(gp_model_ttv, split_indices)

        xlims = [
            (seg[0] - buffer, seg[-1] + buffer) for seg in time_segments if len(seg) > 0
        ]

        bax = brokenaxes(
            xlims=xlims,
            hspace=0.1,
            despine=False,
            subplot_spec=gs[0],
        )

        # Plot each segment with errorbars and model
        for t_seg, d_seg, e_seg, m_seg in zip(
            time_segments, data_segments, error_segments, model_segments
        ):
            if len(t_seg) > 0:
                bax.errorbar(
                    t_seg, d_seg, yerr=e_seg, color="dodgerblue", fmt=".", alpha=0.1
                )
                bax.plot(t_seg, m_seg, color="black", zorder=10)

        # Format broken axes
        bax.set_xlabel("Time (BJD - 2457000)")
        bax.set_ylabel("Relative flux")
        bax.set_ylim([0.98, 1.01])

        plt.ylabel("Relative flux")
        plt.title(f"TTV Fit: {system_name}")  # Add title
        plt.savefig(
            f"{result_path}/transit_lightcurve_ttv_{system_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # O-C plot
        try:
            OC = []
            OC_up_err = []
            OC_down_err = []
            transit_numbers_oc = []  # Redefine locally
            if (
                hasattr(results, "posteriors")
                and "posterior_samples" in results.posteriors
            ):
                post_samples_ttv = results.posteriors["posterior_samples"]
                if "P_p1" in post_samples_ttv and "t0_p1" in post_samples_ttv:
                    P_post = post_samples_ttv["P_p1"]
                    t0_post = post_samples_ttv["t0_p1"]
                    for vt in valid_transits:
                        transit_number = vt["transit_number"]
                        instrument = vt["instrument"]  # Still 'TESS' here
                        ttv_param_name = f"T_p1_{instrument}_{transit_number}"
                        if ttv_param_name in post_samples_ttv:
                            transit_numbers_oc.append(transit_number)
                            computed_time = t0_post + transit_number * P_post
                            observed_time = post_samples_ttv[ttv_param_name]
                            oc_distribution = (
                                (observed_time - computed_time) * 24 * 60.0
                            )
                            val, vup, vdown = juliet.utils.get_quantiles(
                                oc_distribution
                            )
                            OC.append(val)
                            OC_up_err.append(vup - val)
                            OC_down_err.append(val - vdown)

                    if len(transit_numbers_oc) > 0:
                        fig_oc = plt.figure(figsize=(14, 4))
                        plt.errorbar(
                            transit_numbers_oc,
                            OC,
                            yerr=[OC_down_err, OC_up_err],
                            fmt="o",
                            mfc="white",
                            mec="cornflowerblue",
                            ecolor="cornflowerblue",
                            ms=10,
                            elinewidth=1,
                            zorder=3,
                        )
                        # Plot horizontal line only if OC data exists
                        plt.plot(
                            [
                                np.min(transit_numbers_oc) - 0.1,
                                np.max(transit_numbers_oc) + 0.1,
                            ],
                            [0.0, 0],
                            "--",
                            linewidth=1,
                            color="black",
                            zorder=2,
                        )
                        plt.xlim(
                            [
                                np.min(transit_numbers_oc) - 0.1,
                                np.max(transit_numbers_oc) + 0.1,
                            ]
                        )
                        plt.xlabel("Transit number")
                        plt.ylabel("O-C (minutes)")
                        plt.title(f"O-C: {system_name}")  # Add title
                        plt.savefig(
                            f"{result_path}/oc_{system_name}.png",
                            dpi=300,
                            bbox_inches="tight",
                        )
                        plt.close(fig_oc)

                        # Periodogram Analysis
                        OC_days = np.asarray(OC) / (24.0 * 60)
                        OC_up_err_days, OC_down_err_days = (
                            np.asarray(OC_up_err) / (24.0 * 60.0),
                            np.asarray(OC_down_err) / (24.0 * 60.0),
                        )

                        # Add errors in quadrature, LombScargle expects Guassian errors, so need to combine OC_up_err and OC_down_err
                        OC_err_combined = np.sqrt(
                            (OC_up_err_days**2 + OC_down_err_days**2) / 2.0
                        )

                        min_period = 2 * P_post
                        max_period = (time[-1] - time[0]) / 2

                        ls = LombScargle(time, OC_days, OC_err_combined)

                        frequency, power = ls.autopower(
                            minimum_frequency=1 / max_period,
                            maximum_frequency=1 / min_period,
                            samples_per_peak=10,
                        )

        except Exception:
            plt.close("all")
            pass

    else:
        print(f"No valid transits for {system_name}...")
        like_ttv, like_ttv_err = 0, 0

    # Save likelihoods
    delta_logZ = like_ttv - like_no_ttv
    likelihoods = {
        "system_name": system_name,
        "logZ_ttv": like_ttv,
        "logZ_ttv_err": like_ttv_err,
        "logZ_no_ttv": like_no_ttv,
        "logZ_no_ttv_err": like_no_ttv_err,
        "delta_logZ": delta_logZ,
    }

    # Append/Update likelihood CSV using the DataFrame loaded outside the loop
    if system_name in df_likelihoods["system_name"].values:
        # Update existing row
        row_indexer = df_likelihoods["system_name"] == system_name
        for k, v in likelihoods.items():
            df_likelihoods.loc[row_indexer, k] = v
    else:
        # Append new row
        df_likelihoods = pd.concat(
            [df_likelihoods, pd.DataFrame([likelihoods])], ignore_index=True
        )

df_likelihoods.to_csv(like_csv_file, index=False)
