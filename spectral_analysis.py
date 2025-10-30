#!/usr/bin/env python3
"""
Spectral Analysis Script

This script performs a time‑resolved spectral analysis of Fermi‑GBM data,
including:
- loading CSPEC data from all selected NaI (and optionally BGO) detectors,
- estimating and subtracting the background,
- fitting the spectra.

Script options (all are supplied as ``--option value`` on the command line):

--out <filename>
    Name of the CSV file that will receive the fitting results.
    If omitted the script builds a default name of the form:
    ``spectral_evolution_<object>_<t_start>-<t_end>s_<duration>s_duration.csv``.
    (Default: generated automatically as described above.)

--bin-size <size>
    Minimal adaptive bin size (seconds) used when the script divides the
    user‑defined interval into sub‑intervals for time‑resolved fitting.
    The default is **0.25 s** unless the user provides a different value.

--stat <stat>
    Choose the fit statistic: ``cstat`` or ``pgstat``.  Default is ``cstat``.
    
--ignore-bgo
    When present the script excludes the BGO detector (``b0``) from both
    the data loading and response‑matrix loading steps.  By default BGO data
    are **included**.

--use-band
    Forces the spectral fitter to use a single ``Band`` function.
    If this flag is absent the script fits a model consisting of
    ``Comptonized`` + ``BlackBody`` + ``PowerLaw``.  Default behavior is the
    multi‑component model.

--include-errors
    If supplied, failed fits are written to the CSV with NaN values and the
    error message; otherwise only successful fits are saved.  Default is to
    omit failed‑fit entries.

Additional defaults used internally (not exposed as command‑line flags):

* max_beta  = -0.2            # Upper bound for the Band high‑energy index
* min_bin_size = 2           # Default duration per time segment if not overridden
* bkg_fit_degree = 2         # Polynomial order for background fitting
* bin_size_adaptive = True   # Enables adaptive binning in the analysis class
* bkgd_range = [(-50, -10), (30, 100)]   # Background intervals (seconds)
* energy_range_nai = (8, 300)   # NaI detector energy range (keV)
* energy_range_bgo = (325, 9500) # BGO detector energy range (keV)

Usage example:
    $ python spectral_analysis.py --out my_results.csv --bin-size 0.5 \\
        --use-band --include-errors

The script will analyse the interval 1–20 s (hard‑coded in ``main()``),
divide it into 0.5‑s bins, fit each segment with the Band function, and
write the results to *my_results.csv*.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
from gdt.core import data_path
from gdt.missions.fermi.gbm.phaii import GbmPhaii
from gdt.missions.fermi.gbm.collection import GbmDetectorCollection
from gdt.missions.fermi.gbm.response import GbmRsp2
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.core.plot.spectrum import Spectrum
from gdt.core.plot.model import ModelFit
from gdt.core.spectra.fitting import SpectralFitterPgstat, SpectralFitterCstat
from gdt.core.spectra.functions import Band, Comptonized, BlackBody, PowerLaw


class GRBSpectralAnalysis:
    """Class for GRB spectral analysis using Fermi-GBM data"""
    
    def __init__(self, object_no='090926181', **kwargs):
        """Initialize analysis with GRB identifier"""

        self.max_beta = kwargs.get('max_beta', -0.2)
        self.min_bin_size = kwargs.get('min_bin_size', 2)
        self.bkg_fit_degree = kwargs.get('bkg_fit_degree', 2)
        self.bin_size_adaptive = kwargs.get('bin_size_adaptive', True)
        self.stat = kwargs.get('stat', 'cstat')
        
        # Analysis parameters
        self.bkgd_range = [(-50, -10), (30, 100)]  # Background intervals
        self.energy_range_nai = (8, 900)  # NaI energy range (keV)
        self.energy_range_bgo = (325, 9500)  # BGO energy range (keV)

        self.object_no = object_no
        self.object_name = f'bn{object_no}'
        self.setup_paths()
        self.load_data()
        
    def setup_paths(self):
        """Setup file paths for CSPEC and response data"""
        # CSPEC file paths
        common_str = f'datos/{self.object_no}/glg_cspec_'
        self.filepaths_cspec = [
            f"{common_str}n3_{self.object_name}_v00.pha",  # NaI detectors
            f"{common_str}n6_{self.object_name}_v00.pha",
            f"{common_str}n7_{self.object_name}_v00.pha"
        ]
        
        if '--ignore-bgo' not in sys.argv:  # Only include BGO if not ignoring
            self.filepaths_cspec.append(f"{common_str}b0_{self.object_name}_v00.pha")
        
        # Response file paths
        self.filepaths_rsp = [
            f"{common_str}n3_{self.object_name}_v00.rsp2",
            f"{common_str}n6_{self.object_name}_v00.rsp2",
            f"{common_str}n7_{self.object_name}_v00.rsp2"
        ]
        
        if '--ignore-bgo' not in sys.argv:
            self.filepaths_rsp.append(f"{common_str}b0_{self.object_name}_v00.rsp2")
        
    def load_data(self):
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
        
    def fit_background(self):
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
        
        
    def run_time_evolution_analysis(self, start_time=1, end_time=20, **kwargs):
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
        fit_function = Band() if '--use-band' in sys.argv \
            else Comptonized() + BlackBody() + PowerLaw()
            
            
        for i, (name, _, desc) in enumerate(fit_function.param_list):
            if '--use-band' in sys.argv and 'beta' in name.lower():
                fit_function.max_values[i] = max_beta
                fit_function.min_values[i] = -40.0
                break

            if 'blackbody' in name.lower() \
                and 'kt' in name.lower():
                
                fit_function.max_values[i] = 30
                fit_function.min_values[i] = 1E-30

        
        print('PARAMETERS:', fit_function.param_list)
        print('MAX VALUES:', fit_function.max_values)
        print('MIN VALUES:', fit_function.min_values)
        
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
                
            specfitter.fit(fit_function, options={'maxiter': 2000})
            
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
                print(f"✓ Fit successful!")
                results.append(result)
                successful_fits += 1
            else:
                print(f"✗ Fit failed...")
                if t_end < end_time: continue

                failed_fits += 1
                
                if '--include-errors' in sys.argv:
                    results.append(result)
            
            t_start = t_end
            
        # Save results to CSV
        self.save_results_to_csv(results, start_time, end_time, duration)
        
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
        
    def save_results_to_csv(self, results, start_time, end_time, duration):
        """Save spectral fitting results to CSV file"""
        print(f"Saving results to CSV file...")
        
        # Create filename
        filename = \
            f'spectral_evolution_{self.object_name}_{start_time}-{end_time}s_{duration}s_duration.csv'
        filename = get_arg('out', filename)

        # Define CSV headers
        headers = results[0].keys()
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                
                for result in results:
                    writer.writerow(result)
            
            print(f"✓ Results saved to: {filename}")
            
        except Exception as e:
            print(f"✗ Error saving CSV: {e}")

def get_arg(arg_name:str, default:str=None)->str:
    try:
        idx = sys.argv.index(f"--{arg_name}")
        return sys.argv[idx+1]
    except (ValueError, IndexError):
        return default


def main():
    """Main function to run analysis"""
    
    stat = get_arg('stat', 'cstat')
    bin_size = float(get_arg( "bin-size", "0.25"))
    
    # Create analysis instance
    grb_analysis = GRBSpectralAnalysis(

            '090926181'      ### object name
            , min_bin_size=bin_size ### initial value for adaptive bin size
            , bkg_fit_degree=2
            , stat=stat
        
        )
    
    print("\nRunning time evolution analysis (1-20s)...")
    
    # Run time evolution analysis
    results = grb_analysis.run_time_evolution_analysis(start_time=1, end_time=20)
    
    print(f"\nTime evolution analysis complete!")
    print(f"Total time ranges analyzed: {len(results)}")
    print(f"Results saved to CSV file")
    
    return results


if __name__ == "__main__":
    main()
