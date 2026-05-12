# PEDOT:PSS Impedance Prediction Model

This repository provides an image-based analysis pipeline for estimating impedance-related features of printed PEDOT:PSS electrode patterns.

The pipeline detects line boundaries from electrode images, calculates image-derived features, estimates impedance, and computes printed-area-related metrics.

## Image-based impedance estimation for printed PEDOT:PSS ECoG electrodes

<p align="center">
  <img width="940" height="592" alt="image" src="https://github.com/user-attachments/assets/e38d992c-2809-4701-b55c-8ea8d6ec5ba3" />
</p>

<p align="center">
  Schematic overview of the fabrication process and image-based impedance prediction workflow for high-density PEDOT:PSS ECoG electrodes.
</p>

## Usage Conditions

This prediction pipeline was developed for printed PEDOT:PSS electrode patterns under the following conditions:

1. The ink material should be PEDOT:PSS with a concentration of **5 wt% or lower**.
2. The input image should be an optical microscope image captured at **200× magnification**.

The prediction accuracy may decrease if the ink formulation, concentration, imaging magnification, or image quality differs significantly from these conditions.
