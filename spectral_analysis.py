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
from gdt.core.spectra.fitting import SpectralFitterPgstat
from gdt.core.spectra.functions import Band


class GRBSpectralAnalysis:
    """Class for GRB spectral analysis using Fermi-GBM data"""
    
    def __init__(self, object_no='090926181'):
        """Initialize analysis with GRB identifier"""
        self.object_no = object_no
        self.object_name = f'bn{object_no}'
        self.setup_paths()
        self.load_data()
        
    def setup_paths(self):
        """Setup file paths for CSPEC and response data"""
        # CSPEC file paths
        common_str = f'datos/{self.object_no}/glg_cspec_'
        self.filepaths_cspec = [
            f"{common_str}b0_{self.object_name}_v00.pha",  # BGO detector
            f"{common_str}n0_{self.object_name}_v00.pha",  # NaI detectors
            f"{common_str}n1_{self.object_name}_v00.pha",
            f"{common_str}n3_{self.object_name}_v00.pha",
            f"{common_str}n6_{self.object_name}_v00.pha",
            f"{common_str}n7_{self.object_name}_v00.pha"
        ]
        
        # Response file paths
        self.filepaths_rsp = [
            f"{common_str}n3_{self.object_name}_v00.rsp2",
            f"{common_str}n6_{self.object_name}_v00.rsp2",
            f"{common_str}n7_{self.object_name}_v00.rsp2",
            f"{common_str}b0_{self.object_name}_v00.rsp2"
        ]
        
        # Analysis parameters
        self.bkgd_range = [(-50, -10), (30, 100)]  # Background intervals
        self.energy_range_nai = (8, 900)  # NaI energy range (keV)
        self.energy_range_bgo = (325, 35000)  # BGO energy range (keV)
        self.src_range = (1, 2)  # Source interval for spectral analysis
        
    def load_data(self):
        """Load CSPEC and response data"""
        print("Loading CSPEC data...")
        # Load CSPEC data (subset of detectors used in notebook)
        self.cspec_b0 = GbmPhaii.open(self.filepaths_cspec[0])
        self.cspec_n3 = GbmPhaii.open(self.filepaths_cspec[3])
        self.cspec_n6 = GbmPhaii.open(self.filepaths_cspec[4])
        self.cspec_n7 = GbmPhaii.open(self.filepaths_cspec[5])
        
        # Create detector collection
        self.cspecs = GbmDetectorCollection.from_list([
            self.cspec_n3, self.cspec_n6, self.cspec_n7, self.cspec_b0
        ])
        
        print("Loading response matrices...")
        # Load response matrices
        self.rsp_n3 = GbmRsp2.open(self.filepaths_rsp[0])
        self.rsp_n6 = GbmRsp2.open(self.filepaths_rsp[1])
        self.rsp_n7 = GbmRsp2.open(self.filepaths_rsp[2])
        self.rsp_b0 = GbmRsp2.open(self.filepaths_rsp[3])
        
        self.rsps = GbmDetectorCollection.from_list([
            self.rsp_n3, self.rsp_n6, self.rsp_n7, self.rsp_b0
        ])
        
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
        
        # Perform 0th order polynomial fit
        self.backfitters.fit(order=0)
        
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
        
    def extract_spectra(self):
        """Extract source and background spectra"""
        print("Extracting spectra...")
        
        # Extract count spectra
        self.data_specs = self.cspecs.to_spectrum(time_range=self.src_range)
        
        # Extract time-integrated background
        self.bkgd_specs = self.bkgds.integrate_time(*self.src_range)
        
        # Apply energy selection
        self.src_specs = self.cspecs.to_spectrum(
            time_range=self.src_range, 
            nai_kwargs={'energy_range': self.energy_range_nai}, 
            bgo_kwargs={'energy_range': self.energy_range_bgo}
        )
        
        # Convert to PHA format
        self.phas = self.cspecs.to_pha(
            time_ranges=self.src_range, 
            nai_kwargs={'energy_range': self.energy_range_nai}, 
            bgo_kwargs={'energy_range': self.energy_range_bgo}
        )
        
        # Interpolate response matrices at spectrum central time
        self.rsps_interp = [
            rsp.interpolate(pha.tcent) for rsp, pha in zip(self.rsps, self.phas)
        ]
        
    def fit_spectrum(self):
        """Perform spectral fitting with Band function"""
        print("Performing spectral fitting...")
        
        # Initialize spectral fitter
        self.specfitter = SpectralFitterPgstat(
            self.phas, self.bkgds.to_list(), self.rsps_interp, method='TNC'
        )
        
        # Initialize Band function
        self.band = Band()
        
        # Print Band function parameters
        print("Band function parameters:")
        print(f"Parameter list: {self.band.param_list}")
        print(f"Default values: {self.band.default_values}")
        print(f"Min values: {self.band.min_values}")
        print(f"Max values: {self.band.max_values}")
        
        # Perform fit
        print("Fitting Band function...")
        self.specfitter.fit(self.band, options={'maxiter': 1000})
        
        # Display results
        print(f"Fit message: {self.specfitter.message}")
        print(f"Parameters: {self.specfitter.parameters}")
        print(f"90% Asymmetric Errors: {self.specfitter.asymmetric_errors(cl=0.9)}")
        print(f"PGSTAT/DOF: {self.specfitter.statistic}/{self.specfitter.dof}")
        
        return self.specfitter.parameters, self.specfitter.asymmetric_errors(cl=0.9)
        
    def plot_spectrum_fit(self):
        """Plot the fitted spectrum with residuals"""
        print("Plotting spectrum fit...")
        
        # Initialize model plot
        modelplot = ModelFit(fitter=self.specfitter)
        
        # Set axis limits
        ModelFit.hide_residuals(modelplot)
        plt.ylim(1e-4, 200)
        plt.xlim(7.15, 4000)
        ModelFit.show_residuals(modelplot)
        
        # Save plot
        os.makedirs('imagenes', exist_ok=True)
        plt.savefig('imagenes/spectrum_fit.png', bbox_inches='tight')
        plt.show()
        
    def plot_spectra_with_background(self):
        """Plot spectra with background fits and source selections"""
        print("Plotting spectra with background...")
        
        # Plot spectra with background
        specplots = [
            Spectrum(data=data_spec, background=bkgd_spec) 
            for data_spec, bkgd_spec in zip(self.data_specs, self.bkgd_specs)
        ]
        
        for specplot, src_spec in zip(specplots, self.src_specs):
            specplot.add_selection(src_spec)
            
    def fit_multiple_time_ranges(self, start_time=1, end_time=20, duration=2):
        """Iterate over multiple time ranges and fit spectra for each"""
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
        for i in range(start_time, end_time + 1, duration):
            t_start = i
            t_end = i + duration -1
            
            print(f"\n--- Time Range {i-start_time+1}/{end_time-start_time+1}: {t_start}-{t_end}s ---")
            
            try:
                # Define current time range
                current_range = (t_start, t_end)
                
                # Extract spectra for this time range
                data_specs = self.cspecs.to_spectrum(time_range=current_range)
                bkgd_specs = self.bkgds.integrate_time(*current_range)
                
                # Apply energy selection
                src_specs = self.cspecs.to_spectrum(
                    time_range=current_range, 
                    nai_kwargs={'energy_range': self.energy_range_nai}, 
                    bgo_kwargs={'energy_range': self.energy_range_bgo}
                )
                
                # Convert to PHA format
                phas = self.cspecs.to_pha(
                    time_ranges=current_range, 
                    nai_kwargs={'energy_range': self.energy_range_nai}, 
                    bgo_kwargs={'energy_range': self.energy_range_bgo}
                )
                
                # Check if we have sufficient counts
                # total_counts = sum(sum(pha.counts) for pha in phas)
                # print(f"Total counts: {total_counts}")
                
                # if total_counts < 50:  # Minimum counts threshold
                #     print("Skipping - insufficient counts (< 50)")
                #     failed_fits += 1
                #     continue
                
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
                band = Band()
                specfitter.fit(band, options={'maxiter': 1000})
                
                # Get results
                parameters = specfitter.parameters
                errors = specfitter.asymmetric_errors(cl=0.9)
                statistic = specfitter.statistic
                dof = specfitter.dof
                
                # Calculate derived parameter Epeak if not directly fitted
                if len(parameters) >= 2:
                    epeak = parameters[1]  # Epeak is often the second parameter
                else:
                    epeak = np.nan
                
                # Store results
                result = {
                    'time_start': t_start,
                    'time_end': t_end,
                    'time_center': time_center,
                    'duration': duration,
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
                    'pgstat': statistic,
                    'dof': dof,
                    'reduced_pgstat': statistic / dof if dof > 0 else np.nan,
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
                print(f"  PGSTAT/DOF: {result['pgstat']:.1f}/{result['dof']}")
                
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
                    'pgstat': np.nan,
                    'dof': np.nan,
                    'reduced_pgstat': np.nan,
                    'fit_message': str(e),
                    'successful_fit': False
                }
                results.append(result)
        
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
        filename = f'spectral_evolution_{self.object_name}_{start_time}-{end_time}s_{duration}s_duration.csv'
        
        # Define CSV headers
        headers = [
            'time_start', 'time_end', 'time_center', 'duration', 'total_counts',
            'amplitude', 'amplitude_err_low', 'amplitude_err_high',
            'epeak', 'epeak_err_low', 'epeak_err_high',
            'alpha', 'alpha_err_low', 'alpha_err_high',
            'beta', 'beta_err_low', 'beta_err_high',
            'pgstat', 'dof', 'reduced_pgstat', 'fit_message', 'successful_fit'
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
            
    def run_full_analysis(self):
        """Run complete spectral analysis pipeline"""
        print("="*60)
        print("Starting GRB Spectral Analysis")
        print("="*60)
        
        # Fit background
        self.fit_background()
        
        # Extract spectra
        self.extract_spectra()
        
        # Fit spectrum
        parameters, errors = self.fit_spectrum()
        
        # Plot results
        self.plot_spectrum_fit()
        self.plot_spectra_with_background()
        
        print("="*60)
        print("Analysis Complete!")
        print("="*60)
        
        return parameters, errors
        
    def run_time_evolution_analysis(self, start_time=1, end_time=20, duration=2):
        """Run time evolution analysis across multiple time ranges"""
        return self.fit_multiple_time_ranges(start_time, end_time, duration)


def main():
    """Main function to run analysis"""
    # Create analysis instance
    grb_analysis = GRBSpectralAnalysis('090926181')
    
    print("Choose analysis mode:")
    print("1. Single time range analysis (1-2s)")
    print("2. Time evolution analysis (1-20s)")
    
    # For now, run time evolution analysis as requested
    print("\nRunning time evolution analysis (1-20s)...")
    
    # Run time evolution analysis
    results = grb_analysis.run_time_evolution_analysis(start_time=1, end_time=20, duration=2)
    
    print(f"\nTime evolution analysis complete!")
    print(f"Total time ranges analyzed: {len(results)}")
    print(f"Results saved to CSV file")
    
    # Also run single analysis for comparison
    print("\nRunning single time range analysis for comparison...")
    try:
        parameters, errors = grb_analysis.run_full_analysis()
        
        # Print final results
        print("\nSingle Range Spectral Parameters (1-2s):")
        param_names = ['A', 'Epeak', 'alpha', 'beta', 'Epiv']
        for i, (name, param, error) in enumerate(zip(param_names, parameters, errors)):
            if i < len(error):
                print(f"{name}: {param:.3e} (+{error[1]:.3e}, -{error[0]:.3e})")
            else:
                print(f"{name}: {param:.3e}")
    except Exception as e:
        print(f"Single range analysis failed: {e}")
    
    return results


if __name__ == "__main__":
    main()