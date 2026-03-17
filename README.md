# Yield Curve Analysis

This repository contains a Python script and statistical methodology to download, standardise, and analyze US Treasury bond yields representing different maturities over time. This project specifically extracts Principal Components representing level, slope, and curvature from the yield curve and calculates a 1-day Value at Risk (VaR) for a simple bond portfolio.

## Feature Overview

* **Free Public Data Access**: Uses `pandas_datareader` to fetch yield data directly from the Federal Reserve Economic Data (FRED) database. **No API key is required.**
* **Line-by-Line Comments**: The main script (`yield_curve_analysis.py`) is heavily documented. Every single line of code includes a simple English explanation of exactly what it is doing.
* **Statistical Analysis & PCA**:
   * Computes Covariance and Correlation Matrices of yields visually outputting Heatmaps.
   * Performs Z-score standardization on all yield sequences.
   * Implements Principal Component Analysis (PCA) to extract eigenvectors/eigenvalues.
   * Isolates out principal components mapping them statistically to Yield Curve factors such as 'Level', 'Slope' ('Tilt'), and 'Curvature'.
* **Risk Management Calculation**: Uses historical daily percentage variations of 2-year, 5-year, and 10-year term structures alongside their PCA models to calculate the 1-Day 95% Confidence Value at Risk (VaR) of a hypothetical \$5 Million dollar Treasury portfolio.

## Prerequisites

Before running the project, make sure you have installed the required Python libraries. You can use `pip`:

```bash
pip install -r requirements.txt
```

*Note: Ensure you have an active internet connection since the data reader relies on directly contacting FRED web servers.*

## How to Run the Analysis

Since the code renders multiple analytical graphs iteratively, simply run the Python file:

```bash
python yield_curve_analysis.py
```

### Visualizations Outputted
As the script executes, it will sequentially open several Matplotlib windows showing:
1. Covariance Heat Map of standard Treasury Bond Yields.
2. Correlation Heat Map of Treasury Bond Yields.
3. Covariance Heat Map of Standardized Yields (Z-Scores).
4. Time Series Graph of Raw Treasury Yields.
5. Principal Component 1 (Level representation).
6. Calculated 2Y-10Y Tilt graph.
7. Principal Component 2 (Slope representation).
8. Principal Component 3 (Curvature representation).

*You must close each graph window for the script to continue to the next calculation.*

At the final step, the code outputs a calculation estimating the 1-Day 95% VaR for a specifically weighted bond portfolio into the terminal console.

## Project Structure

- `yield_curve_analysis.py` - Main analytical script fetching FRED data and calculating PCA/VaR.
- `requirements.txt` - Required package dependencies.
- `README.md` - Technical overview of the methodology.
