# AstroTemplate

Welcome to Constellation Recognizer! This project aims to automatically detect constellations in images of the night sky using computer vision techniques.

## Overview

Our goal is to detect the 88 constellations defined by the International Astronomical Union (IAU) using OpenCV with Python. The project implements an algorithm that performs the following steps:

1. **Thresholding**: Applies thresholding to the image to black out stars below a certain brightness threshold.
   
2. **Star Detection**: Identifies the 3 brightest stars in the image based on pixel coverage.
   
3. **Triangle Formation**: Forms a triangle using the positions of these stars.

4. **Angle Measurement**: Measures the angles of the triangle formed and stores them as a template for that constellation.

5. **Prediction**: For an unknown image, repeats the above steps and compares the angles with known templates to predict the constellation with minimal angle prediction error.

## Project Structure

The project is organized into the following components:

- **Algorithm Implementation**: Code for constellation detection using OpenCV and Python.
- **Dataset Preparation**: Templates/dataset creation for all 88 constellations.
- **User Interface**: Future work includes creating a UI using Streamlit for easier interaction and visualization.

