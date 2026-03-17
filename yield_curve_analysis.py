# We import pandas to work with data structures like DataFrames
import pandas as pd
# We import pandas_datareader to fetch financial data from different sources, including FRED
import pandas_datareader.data as web
# We import matplotlib.pyplot to draw graphs and visualizations
import matplotlib.pyplot as plt
# We import seaborn to make our plots look nicer and generate heatmaps
import seaborn as sns
# We import numpy for numerical computing and matrix operations
import numpy as np
# We import linear algebra tools from numpy to calculate eigenvectors and eigenvalues
from numpy import linalg as LA

# We apply seaborn's default visual styling to all our matplotlib plots
sns.set_theme()

# We define a list containing the specific FRED codes for Treasury yields of varying maturities
series_ids = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5',
              'DGS7', 'DGS10', 'DGS20', 'DGS30']

# We define a function to pull historical data for one specific Treasury yield series
def get_yield_data(series_id):
    # We use pandas_datareader to fetch data from the 'fred' source within our specific date range
    df = web.DataReader(series_id, 'fred', "1975-01-01", "2024-05-03")
    # We extract and return just the column matching our series_id as a pandas Series
    return df[series_id]

# We create a dictionary by looping over our series_ids list and fetching data for each one
yields_dict = {series_id: get_yield_data(series_id) for series_id in series_ids}

# We convert our dictionary of Series into an integrated two-dimensional DataFrame
yields = pd.DataFrame(yields_dict)

# We rename the columns of our DataFrame to make them easily readable for humans
yields.columns = ['1 Month', '3 Month', '6 Month', '1 Year', '2 Year', '3 Year', '5 Year',
                  '7 Year', '10 Year', '20 Year', '30 Year']

# We explicitly ensure the index (dates) is formatted as proper datetime objects
yields.index = pd.to_datetime(yields.index)

# We remove any rows containing missing data (NaN) so our calculations work perfectly
yields = yields.dropna()

# We calculate the covariance matrix, measuring how much yields move together
covariance_matrix = yields.cov()

# We print a header indicating we are outputting the covariance matrix
print("Covariance Matrix:")

# We print the calculated covariance matrix
print(covariance_matrix)

# We create a new drawing canvas (figure) measuring 8 inches by 6 inches
plt.figure(figsize=(8, 6))

# We plot the covariance matrix as a colored heatmap map, annotating each square with numbers
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt=".1f")

# We set the main title of our covariance heatmap visualization
plt.title('Covariance Heat Map of Treasury Bond Yields')

# We display the plotted heatmap on the screen
plt.show()

# We calculate the correlation matrix, a standardized measure of how yields move together
correlation_matrix = yields.corr()

# We print a header indicating we are outputting the correlation matrix
print("Correlation Matrix:")

# We print the calculated correlation matrix
print(correlation_matrix)

# We create a new 8x6 inch figure for the correlation heatmap
plt.figure(figsize=(8, 6))

# We plot the correlation matrix, displaying numeric annotations up to 2 decimal places
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# We give our correlation heatmap an appropriate title
plt.title('Correlation Heat Map of Treasury Bond Yields')

# We display the correlation heatmap on the screen
plt.show()

# We calculate the arithmetic average (mean) yield for each maturity
yield_means = yields.mean()

# We output a descriptive header for the mean values
print("Yield Means:")

# We show the computed means for all tenors
print(yield_means)

# We calculate the standard deviation for each maturity to measure their historical volatility
yield_stds = yields.std()

# We output a descriptive header for the standard deviations
print("Yield Standard Deviations:")

# We display the standard deviations
print(yield_stds)

# We standardize the data by subtracting the mean and dividing by the standard deviation
standardized_data = (yields - yield_means) / yield_stds

# We announce that we are displaying the top 5 rows of our standardized data
print("Standardized Yield (first 5 rows):")

# We print out the first five records of the resulting standardized DataFrame
print(standardized_data.head())

# We find the new covariance matrix of our fully standardized yield dataset
std_data_cov = standardized_data.cov()

# We instantiate an 8x6 figure for plotting this new covariance matrix
plt.figure(figsize=(8, 6))

# We draw a numerical heatmap of the standardized covariance matrix
sns.heatmap(std_data_cov, annot=True, cmap='coolwarm', fmt=".2f")

# We title our standardized covariance heatmap accordingly
plt.title('Covariance Heat Map of Standardized Treasury Bond Yields')

# We execute the display command to show our plot
plt.show()

# We compute the eigenvalues and eigenvectors from our standardized covariance matrix
eigenvalues, eigenvectors = LA.eig(std_data_cov)

# We print the magnitudes (eigenvalues) of the principal component variances
print(eigenvalues)

# We print the direction vectors (eigenvectors) of the principal components
print(eigenvectors)

# We project our standardized yields onto our eigenvectors to extract Principal Components (PCs)
principal_components = standardized_data.dot(eigenvectors)

# We name the resulting component columns from PC_1 sequentially up to PC_11
principal_components.columns = ["PC_1","PC_2","PC_3","PC_4","PC_5","PC_6","PC_7","PC_8","PC_9","PC_10","PC_11"]

# We display the first 5 records of our principal component time-series data
print(principal_components.head())

# We construct a new DataFrame to neatly organize our sorted eigenvalues using numbers 1 to 11
df_eigval = pd.DataFrame({"Eigenvalues":eigenvalues}, index=range(1,12))

# We calculate the percentage of total variance each eigenvalue/component explains
df_eigval["Explained proportion"] = df_eigval["Eigenvalues"] / np.sum(df_eigval["Eigenvalues"])

# We apply a tabular styling to format 'Explained proportion' numerically as percentages
df_eigval.style.format({"Explained proportion": "{:.2%}"})

# We plot the historical movements of all the raw, unstandardized Treasury Yields together
yields.plot(figsize=(12, 8), title='Figure 2, Treasury Yields', alpha=0.7)

# We move the legend just outside the upper right edge of the plot so it doesn't block lines
plt.legend(bbox_to_anchor=(1.03, 1))

# We render and display the Treasury yield multi-line graph
plt.show()

# We specifically plot the first Principal Component mapping (often interpreted as 'level')
principal_components["PC_1"].plot(figsize=(12, 8), title='Figure 3, Principal Component 1', alpha=0.7)

# We render the PC_1 graph to the screen
plt.show()

# We extract a subset subset from our standardized dataset exactly containing 2-year and 10-year yields
df_s = pd.DataFrame(data = standardized_data)

# We filter the subset columns to only contain '2 Year' and '10 Year' fields
df_s = df_s[["2 Year","10 Year"]]

# We compute the yield curve 'Tilt' or 'Slope' by taking the 2-year yield minus the 10-year yield
df_s["Tilt"] = df_s["2 Year"] - df_s["10 Year"]

# We display the top 5 records of our calculated slope/tilt DataFrame
print(df_s.head())

# We graph the historical movement of our newly calculated 'Tilt' feature
df_s["Tilt"].plot(figsize=(12, 8), title='Figure 4, Tilt of 2-Year Treasury Yield - 10-Year Treasury Yield', alpha=0.7)

# We command matplotlib to draw this plot on the screen
plt.show()

# We graph the second Principal Component representing the curve's slope
principal_components["PC_2"].plot(figsize=(12, 8), title='Figure 5, Principal Component 2', alpha=0.7)

# We display the PC_2 visualization to the user
plt.show()

# We calculate and display the Pearson correlation coefficient between PC_2 and our explicit Tilt metric
print(np.corrcoef(principal_components["PC_2"], df_s["Tilt"]))

# We graph the third Principal Component, which typically captures the curvature of the yield curve
principal_components["PC_3"].plot(figsize=(12, 8), title='Figure 6, Principal Component 3', alpha=0.7)

# We show this PC_3 graph on the screen
plt.show()

# We create an isolated dataset solely comprising 2-year, 5-year, and 10-year bonds
var_dataset = yields[["2 Year","5 Year","10 Year"]]

# We compute the daily percentage return (or drop) metrics across these specific yields
var_yield_chng_dataset = var_dataset.pct_change()

# We drop any NaNs formed precisely on the first day due to percentage change math
var_yield_chng_dataset = var_yield_chng_dataset.dropna()

# We print the starting rows of our yield percentage differences
print(var_yield_chng_dataset.head())

# We find the mean of these daily percentage changes as part of standardization
var_yield_chng_dataset_means = var_yield_chng_dataset.mean()

# We find the standard deviations of these daily percentage changes
var_yield_chng_dataset_stds = var_yield_chng_dataset.std()

# We execute z-score standardization on our daily percentage change table
var_yld_chng_stnd_data = (var_yield_chng_dataset - var_yield_chng_dataset_means) / var_yield_chng_dataset_stds

# We compute the covariance network underlying the standardized daily changes
var_cov_matrix = var_yld_chng_stnd_data.cov()

# We compute new eigenvectors/eigenvalues from this 3-factor covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(var_cov_matrix)

# We define a ranking order by sorting the eigenvalues from largest to smallest variance
sorted_indices = np.argsort(eigenvalues)[::-1]

# We reorder our new eigenvectors strictly according to our sorted magnitude index
pca_components = eigenvectors[:, sorted_indices]

# We assemble our newly sorted eigenvalues into an explanatory 3-row DataFrame
df_eigval = pd.DataFrame({"Eigenvalues":eigenvalues}, index=range(1,4))

# We calculate their percentage of the overall explained mathematical variance
df_eigval["Explained proportion"] = df_eigval["Eigenvalues"] / np.sum(df_eigval["Eigenvalues"])

# We apply styling settings for readability formatting (rendered when using Jupyter)
df_eigval.style.format({"Explained proportion": "{:.2%}"})

# We establish how many top principal components we wish to retain for our models
n_components = 2

# We physically extract just those top 2 explaining eigenvectors
selected_components = pca_components[:, :n_components]

# We define an initial portfolio of bonds assigning a dollar amount holding to specific maturities
portfolio = {
    2: 2000000,  # $2 Million dollars are assigned to 2-year bonds
    5: 2000000,  # $2 Million dollars are assigned to 5-year bonds
    10: 1000000  # $1 Million dollars are assigned to 10-year bonds
}

# We dynamically define the dollar sensitivity by multiplying maturity by our portfolio allocation sizes
sensitivities = np.array([maturity * amount for maturity, amount in portfolio.items()])

# We calculate the day-to-day absolute dollar changes over our retained PCA dimensions
portfolio_changes = (var_yield_chng_dataset * sensitivities) @ selected_components

# We define our target confidence minimum interval to calculate our Value at Risk (VaR)
confidence_level = 0.95

# We compute VaR by taking the 5th percentile extreme loss threshold mathematically
var = -np.percentile(portfolio_changes, 100 * (1 - confidence_level))

# We neatly print the estimated daily worst-case-scenario dollar loss margin (1-Day VaR)
print(f"1-day 95% VaR: ${var:,.2f}")

# We print a basic header for our ultimate calculation summaries
print("\nSummary Statistics:")

# We state the entire total amount invested cumulatively across all bounds in our dictionary
print(f"Portfolio Value: ${sum(portfolio.values()):,.2f}")

# We finally reveal what proportion of our entire portfolio is legally at risk given our 95% threshold
print(f"VaR as % of Portfolio Value: {var / sum(portfolio.values()) * 100:.3f}%")
