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
        chisq_dof = self.backfitters.statistic()[0] / self.backfitters.dof()[0]
        print(f"Background fit quality (chi^2/dof) for first detector: {chisq_dof}")
        
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


def main():
    """Main function to run the analysis"""
    # Create analysis instance
    grb_analysis = GRBSpectralAnalysis('090926181')
    
    # Run full analysis
    parameters, errors = grb_analysis.run_full_analysis()
    
    # Print final results
    print("\nFinal Spectral Parameters:")
    param_names = ['A', 'Epeak', 'alpha', 'beta', 'Epiv']
    for i, (name, param, error) in enumerate(zip(param_names, parameters, errors)):
        if i < len(error):
            print(f"{name}: {param:.3e} (+{error[1]:.3e}, -{error[0]:.3e})")
        else:
            print(f"{name}: {param:.3e}")


if __name__ == "__main__":
    main()