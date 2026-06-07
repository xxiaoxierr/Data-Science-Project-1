### Cross sectional Study on Urban Metrics

This is a capstone project for module Introduction to Data Science. The project can be broken down into two phases: EDA, modelling + product.

### Data

Data is sourced from ATO Database curated by the Asian Development Bank. To simplify analysis between cities, we cleaned original longitudinal data in `data/urban_data_cv.csv` into cross sectional data in `data/all_cleaned.csv` (details in `Data Cleaning.ipynb`). 


### Model

A linear OLS model is tuned to $R^2=0.55$ by using a piecewise linear function to identify breakpoints where variance behaves significantly differently. Such an approach defines change in gradient after the breakpoint / hinge to better model noisy heterogenous data (details in `Model.ipynb`).
$$Access = \beta_0 + \beta_1 RTR + beta_2 QoL + beta_3 max(0, RTR-c)$$


### Data Product

A model that explains 55% variance with 2 variables (with QoL proxy of other measurable data unavailable) is best used as a city benchmark. Model residuals in the upper and lower tails (outliers) are categorised as overperformers and underperformers for city planners to look into what other cities have accomplished. Users may do in-group comparisons for such performers.

The web app comes in 2 tabs, single view and comparison view, as well as a universal sidebar to choose a reference city.  

Take note that 'outliers' here are defined to reflect on their extraordinary public transport access relative to the training data (RTR and QoL). With new data, a new model may discover a different set of 'outliers'. 


### Tech Stack

We employed Python-native frameworks for the entire project, including:
- Streamlit (data product)
- Plotly (data product)
- Altair (data product)

- Pandas 
- NumPy
- Matplotlib
- Scikit learn

