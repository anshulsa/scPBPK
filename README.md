Single-Cell PBPK Models (AZD1775 & Midazolam)

This repository contains Python implementations of physiologically based pharmacokinetic (PBPK) models that include cell-level heterogeneity within organs. Instead of treating tissues as uniform compartments, the models simulate many individual cells along with a bulk population to capture variability in drug exposure across cells.

Two example drugs are included:
	•	AZD1775 (Wee1 inhibitor) — brain model
	•	Midazolam (MDZ) — liver model

Standard Magnolia PBPK model files for these drugs are also included for reference. These represent conventional whole-organ (homogeneous) PBPK dynamics and are provided for comparison with the single-cell models.


Key Features

	•	Explicit simulation of heterogeneous single cells
	•	Bulk compartment representing remaining cells
	•	Conserved total organ capacity
	•	Cell-to-cell variability in transport and/or metabolism
	•	Tracking of intracellular drug concentrations


Requirements
	•	Python 3.x
	•	NumPy
	•	SciPy
	•	Pandas
	•	Matplotlib

