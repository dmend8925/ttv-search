# Preliminary check for systems in systems.csv, will create plots of all the available
# lightcurves. You can then choose to either use all available lightcurves or make a list
# that picks out the ones that you want in the main script.
# Python 3.8.2
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os

csv_filename = "systems.csv"
output_plot_dir = "sector_lightcurve_plots"
os.makedirs(output_plot_dir, exist_ok=True)

# Read System Names
data = pd.read_csv(csv_filename, comment="#")
system_names = data["system_name"].dropna().unique().tolist()

# Loop Through Systems
for system_name in system_names:
    print(f"\nProcessing: {system_name}")
    try:
        search_result = lk.search_lightcurve(system_name, mission="TESS", author="SPOC")

        if not search_result:
            print(f"  No suitable light curves found for {system_name}.")
            continue

        print(f"  Found {len(search_result)} light curve products. Downloading all...")
        lc_collection_raw = search_result.download_all(quality_bitmask="default")

        if not lc_collection_raw:
            print(f"  Failed to download any light curves for {system_name}.")
            continue

        # Filter out empty light curves before processing loop
        valid_lcs_for_plotting = []
        print("  Filtering downloaded light curves...")
        for lc_individual in lc_collection_raw:
            if (
                hasattr(lc_individual, "time")
                and lc_individual.time is not None
                and len(lc_individual.time) > 0
            ):
                valid_lcs_for_plotting.append(lc_individual)
            else:
                sector = lc_individual.meta.get("SECTOR", "N/A")
                author = lc_individual.meta.get(
                    "AUTHOR", lc_individual.meta.get("ORIGIN", "Unknown")
                )
                print(
                    f"    Skipping downloaded LC (Sector {sector}, {author}): No time data or empty."
                )

        if not valid_lcs_for_plotting:
            print(
                f"  No valid light curves with data found after initial check for {system_name}."
            )
            continue

        n_lcs = len(valid_lcs_for_plotting)
        print(f"  Processing {n_lcs} valid light curves...")

        # Determine subplot layout
        ncols = 3
        nrows = math.ceil(n_lcs / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 5, nrows * 3),
            sharey=True,
            squeeze=False,
            constrained_layout=True,
        )
        axes = axes.flat

        # Iterate through light curve
        plot_count = 0
        processed_lcs_info = []
        for i, lc_raw in enumerate(valid_lcs_for_plotting):
            mission = lc_raw.meta.get("MISSION", "Unknown")
            sector = lc_raw.meta.get("SECTOR", "N/A")
            author = lc_raw.meta.get("AUTHOR", lc_raw.meta.get("ORIGIN", "Unknown"))
            label = f"{mission} S{sector} ({author})"

            print(f"    Processing {label}...")
            try:
                # Process this individual LC
                lc_step1 = lc_raw.remove_nans()
                if lc_step1 is None or len(lc_step1.time) == 0:
                    continue
                try:
                    lc_step2 = lc_step1.remove_outliers(sigma=5)
                    if lc_step2 is None or len(lc_step2.time) == 0:
                        lc_step2 = lc_step1  # Fallback
                except Exception:
                    lc_step2 = lc_step1  # Fallback

                lc_processed = lc_step2.normalize()
                if lc_processed is None or len(lc_processed.time) == 0:
                    continue

                time_val = lc_processed.time.value
                flux_val = lc_processed.flux.value
                # Check for flux_err existence before accessing .value
                if (
                    hasattr(lc_processed, "flux_err")
                    and lc_processed.flux_err is not None
                ):
                    flux_err_val = lc_processed.flux_err.value
                else:
                    flux_err_val = np.full_like(
                        time_val, np.nan
                    )  # Assign NaNs if missing

                # Ensure conversion to standard numpy arrays
                t = np.asarray(time_val)
                f = np.asarray(flux_val)
                ferr = np.asarray(flux_err_val)

                # Ensure consistent lengths (should be handled by lightkurve)
                min_len = min(len(t), len(f), len(ferr))
                t, f, ferr = t[:min_len], f[:min_len], ferr[:min_len]

                # Remove rows where time or flux are not finite
                finite_data_mask = np.isfinite(t) & np.isfinite(f)
                t = t[finite_data_mask]
                f = f[finite_data_mask]
                ferr = ferr[finite_data_mask]

                invalid_err_mask = ~np.isfinite(ferr) | (ferr <= 0)
                ferr[invalid_err_mask] = np.nan

                if len(t) < 10:
                    print(
                        "      Skipping: Too few valid/finite points after final processing."
                    )
                    continue

                # Plotting this sector's data
                if plot_count < len(axes):
                    ax = axes[plot_count]

                    # Matplotlib's errorbar handles NaNs in yerr by not plotting the bar
                    ax.errorbar(
                        t,
                        f,
                        yerr=ferr,
                        fmt=".",
                        markersize=2,
                        alpha=0.6,
                        elinewidth=0.5,
                        ecolor="grey",
                        color="black",
                        capsize=0,
                    )

                    ax.set_title(label, fontsize=10)
                    ax.set_xlabel(
                        f"Time ({lc_processed.time.format.upper()})", fontsize=9
                    )
                    ax.set_xlim(t.min(), t.max())
                    ax.grid(alpha=0.3)
                    if plot_count % ncols == 0:
                        ax.set_ylabel("Norm. Flux", fontsize=9)
                    processed_lcs_info.append(
                        {"label": label, "time": t, "flux": f, "error": ferr}
                    )
                    plot_count += 1
                else:
                    print("      Warning: Ran out of plot axes unexpectedly.")
                    break

            except Exception as e_proc:
                print(f"Error processing individual LC {label}: {e_proc}")
                continue

        # Clean up unused axes
        for i in range(plot_count, len(axes)):
            axes[i].set_visible(False)

        # Add overall figure title and adjust layout
        if plot_count > 0:
            fig.suptitle(f"Individual Sector Light Curves: {system_name}", fontsize=16)

            # Save the combined figure
            plot_filename = os.path.join(output_plot_dir, f"sectors_{system_name}.png")
            plt.savefig(plot_filename, dpi=200, bbox_inches="tight")
            print(f"  Saved sector plot: {plot_filename}")
        else:
            print(f"  No valid sectors plotted for {system_name}.")

        plt.close(fig)

    except Exception as e:
        print(f"Error processing system {system_name}: {e}")
        plt.close("all")
        continue
