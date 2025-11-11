#!/usr/bin/env python3
# Use `--help` flag on this script to see info and usage.

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Any, Literal

from gdt.missions.fermi.gbm.phaii import GbmPhaii
from gdt.missions.fermi.gbm.collection import GbmDetectorCollection
from gdt.missions.fermi.gbm.response import GbmRsp2
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.core.spectra.fitting import SpectralFitterPgstat, SpectralFitterCstat
from gdt.core.spectra.functions import Band, Comptonized, BlackBody, PowerLaw
from gdt.core.plot.lightcurve import Lightcurve


class GRBSpectralAnalysis:
    """Class for GRB spectral analysis using Fermi-GBM data"""

    def __init__(
            self, obj: str, fit_type: Literal["band", "multi"],
            ignore_bgo: bool, include_errors: bool,
            stat: str, out: str,
            max_beta: float = -0.2,
            min_bin_size: float = 2,
            bin_size_adaptive: bool = True,
            bkg_fit_degree: int = 2,
            ) -> None:
        """Initialize analysis with GRB identifier"""
        # This exposes all current arguments to class attributes.
        for key, value in locals().items():
            setattr(self, key, value)

        # Analysis parameters
        self.bkgd_range: list[tuple[float, float]] = [(-50, -10), (30, 100)]  # Background intervals
        self.energy_range_nai: tuple[float, float] = (8, 900)  # NaI energy range (keV)
        self.energy_range_bgo: tuple[float, float] = (325, 9500)  # BGO energy range (keV)

        self.object_name: str = f'bn{obj}'
        self.filepaths_cspec: list[str] = []
        self.filepaths_rsp: list[str] = []
        self.setup_paths()
        self.load_data()

    def setup_paths(self) -> None:
        """Setup file paths for CSPEC and response data"""
        # CSPEC file paths
        common_str = f'datos/{self.obj}/glg_cspec_'

        if self.obj=='090926181':
            dets = ('n3','n6','n7','b0')
        elif self.obj=='090424592':
            dets = ('n7','n8','nb','b1')

        assert dets, "No se han encotrado datos del objeto..."
        all_paths = [
            [ f"{common_str}{dt}_{self.object_name}_v00.{ext}" for dt in dets] \
            for ext in ('pha', 'rsp2')
        ]

        self.filepaths_cspec, self.filepaths_rsp = all_paths
        if self.ignore_bgo:
            self.filepaths_cspec.pop(-1)
            self.filepaths_rsp.pop(-1)

    def load_data(self) -> None:
        """Load CSPEC and response data"""
        print("Loading CSPEC data...")
        self.cspecs = GbmDetectorCollection.from_list(
            [ GbmPhaii.open(p) for p in self.filepaths_cspec ]
            )

        print("Loading response matrices...")
        self.rsps = GbmDetectorCollection.from_list(
            [ GbmRsp2.open(p) for p in self.filepaths_rsp ]
        )

        print("Data loading complete!")

    def fit_background(self) -> None:
        """Fit background using polynomial models"""
        print("Fitting background...")

        # Initialize background fitters for each detector
        self.backfitters = [
            BackgroundFitter.from_phaii(cspec, Polynomial, time_ranges=self.bkgd_range)
            for cspec in self.cspecs
        ]
        self.backfitters = GbmDetectorCollection.from_list(
            self.backfitters, dets=self.cspecs.detector()
        )

        # Perform nth order polynomial fit
        self.backfitters.fit(order=self.bkg_fit_degree)

        # Calculate chi-squared for quality assessment
        # chisq_dof = self.backfitters.statistic()[0] / self.backfitters.dof()[0]
        # print(f"Background fit quality (chi^2/dof) for first detector: {chisq_dof}")

        # Interpolate background fits
        self.bkgds = self.backfitters.interpolate_bins(
            self.cspecs.data()[0].tstart,
            self.cspecs.data()[0].tstop
        )
        self.bkgds = GbmDetectorCollection.from_list(
            self.bkgds, dets=self.cspecs.detector()
        )


    def run_time_evolution_analysis(self, start_time: float, end_time: float, **kwargs: Any) -> list[dict[str, Any]]:
        """Iterate over multiple time ranges and fit spectra for each"""

        assert 0 < start_time < end_time, "Invalid time range"

        max_beta = kwargs.get('max_beta', self.max_beta)
        duration = kwargs.get('duration', self.min_bin_size)

        print("="*60)
        print("MULTIPLE TIME RANGE SPECTRAL ANALYSIS")
        print("="*60)
        print(f'USING_STAT: {self.stat}')
        print(f"Analyzing time ranges from {start_time} to {end_time} seconds")
        print(f"Duration per range: {duration} seconds")
        print(f"Total ranges: {end_time - start_time + 1}")
        print("="*60)

        # Initialize results storage
        results = []
        successful_fits = 0
        failed_fits = 0

        # Ensure background is fitted
        if not hasattr(self, 'bkgds'):
            print("Fitting background first...")
            self.fit_background()

        # Initialize and fit Band function
        fit_function = Band() if self.fit_type=="band"\
            else Comptonized() + BlackBody() + PowerLaw()

        for i, (name, _, desc) in enumerate(fit_function.param_list):
            if self.fit_type=="band":
                if 'beta' in name.lower():
                    fit_function.max_values[i] = max_beta
                    fit_function.min_values[i] = -40.0
                    break
                continue

            if desc=='Temperature':
                fit_function.max_values[i] = 50
                fit_function.min_values[i] = 1E-80

            elif desc == 'Amplitude':
                fit_function.min_values[i] = -1.0

            elif desc == 'Photon index':
                fit_function.max_values[i] = 30
                fit_function.min_values[i] = -2.5

        print('VALORES_DE_PARAMETROS:')
        for idx, (par,min,max) in enumerate(zip(
            fit_function.param_list,
            fit_function.min_values,
            fit_function.max_values
        )):
            print(idx,par,min,max)

        # Iterate over time ranges
        t_start:float = start_time
        t_end:float = start_time

        while t_start < end_time:
            t_end += duration
            print(f"\n--- Time Range: {t_start}-{t_end}s ---")

            # Define current time range
            current_range = (t_start, t_end)
            # Extract spectra for this time range
            # data_specs = self.cspecs.to_spectrum(time_range=current_range)
            # bkgd_specs = self.bkgds.integrate_time(*current_range)

            # Apply energy selection
            # src_specs = self.cspecs.to_spectrum(
            #     time_range=current_range,
            #     nai_kwargs={'energy_range': self.energy_range_nai},
            #     bgo_kwargs={'energy_range': self.energy_range_bgo}
            # )

            # Convert to PHA format
            phas = self.cspecs.to_pha(
                time_ranges=current_range,
                nai_kwargs={'energy_range': self.energy_range_nai},
                bgo_kwargs={'energy_range': self.energy_range_bgo}
            )

            # Interpolate response matrices at time center
            time_center = (t_start + t_end) / 2
            rsps_interp = [
                rsp.interpolate(time_center) for rsp in self.rsps
            ]

            # Initialize spectral fitter
            specfitter_funct = SpectralFitterPgstat \
                if self.stat=='pgstat' else SpectralFitterCstat
            specfitter = specfitter_funct(
                phas, self.bkgds.to_list(), rsps_interp, method='TNC'
            )
            specfitter.fit(fit_function, options={'maxiter': 10000})

            # Get results
            parameters = specfitter.parameters
            errors = specfitter.asymmetric_errors(cl=0.9)
            statistic = specfitter.statistic
            dof = specfitter.dof
            epeak_err = errors[1][1]
            for i, (name, _, desc) in enumerate(fit_function.param_list):
                is_epeak = 'SED PEAK' in desc.upper()
                if not is_epeak: continue
                epeak_err = errors[i][1]
            fit_success = epeak_err < np.inf

            # Store results
            result = {
                'time_start': t_start,
                'time_end': t_end,
                'time_center': time_center,
                'duration': t_end - t_start,
                'stat': statistic if fit_success else np.nan,
                'dof': dof if fit_success else np.nan,
                'reduced_stat': statistic / dof if dof > 0 else np.nan,
                'fit_message': specfitter.message,
                'successful_fit': fit_success
            }

            for (name, _, _), value, (low, high) in \
                zip(fit_function.param_list, parameters, errors):
                result[f'{name}'] = value
                result[f'{name}_err_low'] = low
                result[f'{name}_err_high'] = high

            if fit_success:
                print(f"✔ Fit successful!")
                results.append(result)
                successful_fits += 1
            else:
                print(f"✗ Fit failed...")
                if t_end < end_time: continue
                failed_fits += 1
                if self.include_errors:
                    results.append(result)

            t_start = t_end

        # Save results to CSV
        self.out = self.out if self.out != 'generated' else f'spectral_evolution_{self.obj}_{start_time}-{end_time}s_{duration}s_duration.csv'
        self.save_results_to_csv(results)

        # Print summary
        print("\n" + "="*60)
        print("MULTIPLE TIME RANGE ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Successful fits: {successful_fits}")
        print(f"Failed fits: {failed_fits}")
        print(f"Total ranges analyzed: {len(results)}")
        print(f"Results saved to CSV file")
        print("="*60)

        return results

    def graph(self, results: list[dict[str, Any]], columns: list[str], show: bool) -> None:
        """Creates and shows graphs of given results."""
        # Read the CSV file
        df: pd.DataFrame = pd.DataFrame(results)
        # Create figure and axis
        cspec_obj=GbmPhaii.open(f'datos/{self.obj}/glg_cspec_b{int('592' in self.obj)}_bn{self.obj}_v00.pha')
        t_range = (0,18)
        e_range = (325,9500)
        lc_data = cspec_obj.to_lightcurve(time_range=t_range, energy_range=e_range)
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True, gridspec_kw={'hspace': 0})
        _ = Lightcurve(data=lc_data, ax=ax1)
        axes =(ax1, ax2, ax3, ax4)
        colors = ('blue','orange','green', 'red' )
        scales = ('linear', 'linear', 'linear', 'linear')
        for idx, (ax, param, color, scale) in enumerate(zip(axes, columns, colors, scales)):
            if idx > 0:
                ax.errorbar(df['time_center'], df[param],
                    yerr=[df[f'{param}_err_low']
                    , df[f'{param}_err_high']],
                         fmt='.-', color=color, capsize=3, label=param)
                ax.set_ylabel(param)
            ax.set_yscale(scale)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle('Spectral Parameters Evolution with Error Bars')
        plt.tight_layout()
        plt.savefig('imagenes/grafica.png')
        if show: plt.show()
        return

    def save_results_to_csv(self, results: list[dict[str, Any]]) -> None:
        """Save spectral fitting results to CSV file"""
        print("Saving results to CSV file...")
        # Define CSV headers.
        headers = results[0].keys()
        with open(self.out, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"✔ Results saved to: {self.out}")


def get_cmd_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog="SpectralAnalysis",
        formatter_class=RawDescriptionHelpFormatter,
        description="""This script performs a time‑resolved spectral analysis
of Fermi‑GBM data, including:

    - loading CSPEC data from all selected NaI (and optionally BGO) detectors,
    - estimating and subtracting the background,
    - fitting the spectra.""",
        epilog="""Additional defaults used internally (not exposed as command‑line flags):

    - max_beta  = -0.2            # Upper bound for the Band high‑energy index
    - min_bin_size = 2           # Default duration per time segment if not overridden
    - bkg_fit_degree = 2         # Polynomial order for background fitting
    - bin_size_adaptive = True   # Enables adaptive binning in the analysis class
    - bkgd_range = [(-50, -10), (30, 100)]   # Background intervals (seconds)
    - energy_range_nai = (8, 300)   # NaI detector energy range (keV)
    - energy_range_bgo = (325, 9500) # BGO detector energy range (keV)

The script will analyse the interval, divide it bins, fit each segment with the Band function, and write the results
to *my_results.csv*.""")
    parser.add_argument(
        "--obj", type=str, default="090926181",
        help="""GRB object identifier to analyze. Supported identifiers are `090926181` and
        `090424592`. This determines which detector data and response files are loaded.""")
    parser.add_argument(
        "--out", type=str, default="generated",
        help="""Name of the CSV file that will receive the fitting results.
    The "generated" value builds a name of the form:
    `spectral_evolution_<object>_<t_start>-<t_end>s_<duration>s_duration.csv`.""")
    parser.add_argument("--bin-size", type=float, default=0.25,
        help="""Minimal adaptive bin size (seconds) used when the script divides the
    user‑defined interval into sub‑intervals for time‑resolved fitting.
    The default is 0.25 unless the user provides a different value.""")
    parser.add_argument("--stat", choices=["cstat", "pgstat"], default="cstat",
        help="""Choose the fit statistic: ``cstat`` or ``pgstat``.  Default is ``cstat``.""")
    parser.add_argument("--ignore-bgo", action="store_true",
        help="""When present the script excludes the BGO detector (``b0``) from both
    the data loading and response‑matrix loading steps.""")
    parser.add_argument("--fit-func", choices=["band", "multi"], default="multi",
        help="""The function to use the spectral fitter with. `band` option
    uses the `Band` function and if `multi` it uses the sum
    of `Comptonized`, `BlackBody` and `PowerLaw`. Default is `multi`.""")
    parser.add_argument("--include-errors", action="store_true",
        help="""If supplied, failed fits are written to the CSV with NaN values and the
    error message; otherwise only successful fits are saved.""")
    parser.add_argument("--show", action="store_true",
        help="""Whether to show or not the graphs created.""")
    args: Namespace = parser.parse_args()
    return args

def main() -> list[dict[str, Any]]:
    """Main function to run analysis"""

    start_time:int = 1
    end_time:int = 17
    args: Namespace = get_cmd_args()

    # Create analysis instance.
    grb_analysis = GRBSpectralAnalysis(
            args.obj,      ### object name
            out=args.out,
            min_bin_size=args.bin_size, ### initial value for adaptive bin size
            bkg_fit_degree=2,
            stat=args.stat,
            fit_type=args.fit_func,
            ignore_bgo=args.ignore_bgo,
            include_errors=args.include_errors)
    
    print("\nRunning time evolution analysis ({start_time}-{end_time}s)...")
    results = grb_analysis.run_time_evolution_analysis(start_time=start_time, end_time=end_time)
    
    print(f"\nTime evolution analysis complete!")
    print(f"Total time ranges analyzed: {len(results)}")
    print(f"Results saved to CSV file")
    
    # Create columns depending on what fit function it's in use.
    columns: list[str] = ['A','Comptonized: Epeak','BlackBody: kT','PowerLaw: A']
    if args.fit_func == "band":
        columns = ['A','Epeak','alpha','beta']
    grb_analysis.graph(results, columns, show=args.show)
    
    return results

if __name__ == "__main__":
    main()
