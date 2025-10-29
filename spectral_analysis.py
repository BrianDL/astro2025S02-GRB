#!/usr/bin/env python3
"""
Spectral Analysis Script for GRB 090926181
Based on analisis_espectral_multiple.ipynb

This script performs spectral analysis of GRB data including:
- Data loading from multiple detectors
- Background estimation and subtraction
- Spectral fitting with Band function
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
        
        # Analysis parameters
        self.bkgd_range = [(-50, -10), (30, 100)]  # Background intervals
        self.energy_range_nai = (8, 300)  # NaI energy range (keV)
        self.energy_range_bgo = (325, 9500)  # BGO energy range (keV)
        self.src_range = (1, 2)  # Source interval for spectral analysis
        
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
        
        
    def fit_multiple_time_ranges(self, start_time=1, end_time=20, **kwargs):
        """Iterate over multiple time ranges and fit spectra for each"""

        assert 0 < start_time < end_time, "Invalid time range"

        max_beta = kwargs.get('max_beta', self.max_beta)
        duration = kwargs.get('duration', self.min_bin_size)

        print("="*60)
        print("MULTIPLE TIME RANGE SPECTRAL ANALYSIS")
        print("="*60)
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
        
        # Iterate over time ranges
        t_start:float = start_time
        t_end:float = start_time

        while t_start < end_time:
            t_end += duration
            
            print(f"\n--- Time Range: {t_start}-{t_end}s ---")
            
            # Define current time range
            current_range = (t_start, t_end)
            
            # Extract spectra for this time range
            data_specs = self.cspecs.to_spectrum(time_range=current_range)
            bkgd_specs = self.bkgds.integrate_time(*current_range)
            
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
            specfitter = SpectralFitterPgstat(
                phas, self.bkgds.to_list(), rsps_interp, method='TNC'
            )
            
            # Initialize and fit Band function
            fit_function = Band() if '--use-band' in sys.argv \
                else Comptonized() + BlackBody() + PowerLaw()
            
            fit_function.max_values[3] = max_beta
                
            try:
                specfitter.fit(fit_function, options={'maxiter': 2000})
                
                # Get results
                parameters = specfitter.parameters
                errors = specfitter.asymmetric_errors(cl=0.9)
                statistic = specfitter.statistic
                dof = specfitter.dof
                
                # Calculate derived parameter Epeak if not directly fitted
                if len(parameters) >= 2:
                    epeak = parameters[1]
                else:
                    epeak = np.nan
                
                if len(errors) > 1 and len(errors[1]) > 1:
                    epeak_err = errors[1][1]

                if epeak_err == np.inf:
                    print("ERRROR_HIGH:", epeak_err)
                    if t_end >= end_time:
                        raise ValueError("Fit Not Found in the rest of the interval")

                    continue

                # Store results
                result = {
                    'time_start': t_start,
                    'time_end': t_end,
                    'time_center': time_center,
                    'duration': t_end - t_start,
                    'total_counts': 0, ### fix: total_counts,
                    'amplitude': parameters[0] if len(parameters) > 0 else np.nan,
                    'amplitude_err_low': errors[0][0] if len(errors) > 0 and len(errors[0]) > 0 else np.nan,
                    'amplitude_err_high': errors[0][1] if len(errors) > 0 and len(errors[0]) > 1 else np.nan,
                    'epeak': epeak,
                    'epeak_err_low': errors[1][0] if len(errors) > 1 and len(errors[1]) > 0 else np.nan,
                    'epeak_err_high': errors[1][1] if len(errors) > 1 and len(errors[1]) > 1 else np.nan,
                    'alpha': parameters[2] if len(parameters) > 2 else np.nan,
                    'alpha_err_low': errors[2][0] if len(errors) > 2 and len(errors[2]) > 0 else np.nan,
                    'alpha_err_high': errors[2][1] if len(errors) > 2 and len(errors[2]) > 1 else np.nan,
                    'beta': parameters[3] if len(parameters) > 3 else np.nan,
                    'beta_err_low': errors[3][0] if len(errors) > 3 and len(errors[3]) > 0 else np.nan,
                    'beta_err_high': errors[3][1] if len(errors) > 3 and len(errors[3]) > 1 else np.nan,
                    'stat': statistic,
                    'dof': dof,
                    'reduced_stat': statistic / dof if dof > 0 else np.nan,
                    'fit_message': specfitter.message,
                    'successful_fit': True
                }
                
                results.append(result)
                successful_fits += 1
                
                # Print results for this time range
                print(f"✓ Fit successful!")
                print(f"  Amplitude: {result['amplitude']:.3e}")
                print(f"  Epeak: {result['epeak']:.1f} keV")
                print(f"  Alpha: {result['alpha']:.2f}")
                print(f"  Beta: {result['beta']:.2f}")
                print(f"  STAT/DOF: {result['stat']:.1f}/{result['dof']}")
                
            except Exception as e:
                print(f"✗ Fit failed: {e}")
                failed_fits += 1
                
                # Store failed result
                result = {
                    'time_start': t_start,
                    'time_end': t_end,
                    'time_center': (t_start + t_end) / 2,
                    'duration': duration,
                    'total_counts': np.nan,
                    'amplitude': np.nan,
                    'amplitude_err_low': np.nan,
                    'amplitude_err_high': np.nan,
                    'epeak': np.nan,
                    'epeak_err_low': np.nan,
                    'epeak_err_high': np.nan,
                    'alpha': np.nan,
                    'alpha_err_low': np.nan,
                    'alpha_err_high': np.nan,
                    'beta': np.nan,
                    'beta_err_low': np.nan,
                    'beta_err_high': np.nan,
                    'stat': np.nan,
                    'dof': np.nan,
                    'reduced_stat': np.nan,
                    'fit_message': str(e),
                    'successful_fit': False
                }
                
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
        headers = [
            'time_start', 'time_end', 'time_center', 'duration', 'total_counts',
            'amplitude', 'amplitude_err_low', 'amplitude_err_high',
            'epeak', 'epeak_err_low', 'epeak_err_high',
            'alpha', 'alpha_err_low', 'alpha_err_high',
            'beta', 'beta_err_low', 'beta_err_high',
            'stat', 'dof', 'reduced_stat', 'fit_message', 'successful_fit'
        ]
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                
                for result in results:
                    writer.writerow(result)
            
            print(f"✓ Results saved to: {filename}")
            
        except Exception as e:
            print(f"✗ Error saving CSV: {e}")
            
    def run_time_evolution_analysis(self, start_time=1, end_time=20, **kwargs):
        """Run time evolution analysis across multiple time ranges"""
        duration = kwargs.get('duration', self.min_bin_size)
        return self.fit_multiple_time_ranges(start_time, end_time, duration=duration)


def get_arg(arg_name:str, default:str=None)->str:
    try:
        idx = sys.argv.index(f"--{arg_name}")
        return sys.argv[idx+1]
    except (ValueError, IndexError):
        return default


def main():
    """Main function to run analysis"""
    
    bin_size = float(get_arg( "bin-size", "0.25"))
    
    # Create analysis instance
    grb_analysis = GRBSpectralAnalysis(

            '090926181'      ### object name
            , min_bin_size=bin_size ### initial value for adaptive bin size
            , bkg_fit_degree=2
        
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