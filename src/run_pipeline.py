from src.preprocessing import load_data, clean_data, rename_columns, standardise, add_breakpoint, prepare_features, save_unit_dict, save_country_name_dict
from src.model import fit_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer

from pathlib import Path
import json

# 1. Load absolute path of data
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "all_cleaned.csv"

data = load_data(DATA_PATH)
data = rename_columns(data)
cleaned = clean_data(data)

data = clean_data(data, returnall=True)

# 2. Run model fitting pipeline
def run_pipeline(cleaned):
    std_data = standardise(cleaned)
    std_data = add_breakpoint(std_data)

    X, y = prepare_features(std_data)
    model = fit_model(X, y)
    
    return model, std_data

def process_outliers(data):
    overperformer_pct = data[data['stud_residual'] > 
                                np.percentile(data['stud_residual'], 90)]

    underperformer_pct = data[data['stud_residual'] <
                                np.percentile(data['stud_residual'], 10)]

    overperformer = overperformer_pct.index.tolist()
    underperformer = underperformer_pct.index.tolist()
    return overperformer, underperformer

# 3. Process and save outputs
def process_outputs(model, data):
    processed = data.copy()

    processed['fitted'] = model.fittedvalues
    processed['residual'] = model.resid
    processed['stud_residual'] = model.get_influence().resid_studentized_external

    return processed

def save_outputs(processed):
    overperformer, underperformer = process_outliers(processed)

    # Save to JSON
    try:
        with open('data/outliers.json', 'w') as f:
            json.dump({
                'overperformer': overperformer,
                'underperformer': underperformer}, f)
    except Exception as e:
        print ("Error: ", e)
        return False

    # Save outputs to parquet
    try:
        processed.to_parquet("outputs/model_results.parquet", index=True)
        processed.loc[overperformer, :].to_parquet("outputs/model_results_overperformer", index=True)
        processed.loc[underperformer, :].to_parquet("outputs/model_results_underperformer", index=True)
        return True
    
    except Exception as e:
        print ("Error: ", e)
        return False
    
# 4. Variable selection for spider chart
def get_outliers():
    with open("data/outliers.json", "r") as f:
        outliers = json.load(f)

    return outliers["overperformer"], outliers["underperformer"]

def get_numeric_col(data):
    return data.select_dtypes(include='number').columns.tolist()

    
def spider_chart_variables(data):
    # Numeric columns only
    numeric_cols = get_numeric_col(data)
    overperformer, underperformer = get_outliers()

    # For overperformers and underperformers
    over_df = data.loc[overperformer, numeric_cols]
    under_df = data.loc[underperformer, numeric_cols]

    # Count nan
    nan_per_column_over = over_df.isna().sum()
    nan_per_column_under = under_df.isna().sum()

    to_use_over = nan_per_column_over[nan_per_column_over < 10].index.tolist()
    to_use_under = nan_per_column_under[nan_per_column_under < 10].index.tolist()

    return list(set(to_use_over) & set(to_use_under))

def save_spider_chart_variables(variables):
    urban_form, pct_metrics, city_iden, urban_mobility = get_dashboard_variables() 
    
    try:
        with open('data/spider_chart_variables.json', 'w') as f:
            json.dump({'variables': variables,
                       'to_use_variables': urban_mobility}, f)
        return True
    
    except Exception as e:
        print ("Error: ", e)
        return False

def get_spider_chart_variables():
    with open("data/spider_chart_variables.json", "r") as f:
        var = json.load(f)

    return var['to_use_variables']

def preprocess_spider_chart(df):
    numeric_col = get_numeric_col(df)
    features = df[numeric_col]

    imputer = KNNImputer(n_neighbors=2, weights='uniform')
    imputed_array = imputer.fit_transform(features)

    df_imputed = pd.DataFrame(imputed_array, columns=features.columns, index=df.index)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_imputed)
    scaled_data = pd.DataFrame(
        scaled_data,
        columns=features.columns,
        index=df.index
    )

    return df_imputed, scaled_data

def save_spider_chart_data(data):
    overperformer, underperformer = get_outliers()
    var = get_spider_chart_variables()
    
    all_df, all_df_scaled = preprocess_spider_chart(data.loc[:, var])
    over_df, over_df_scaled = preprocess_spider_chart(data.loc[overperformer, var])
    under_df, under_df_scaled = preprocess_spider_chart(data.loc[underperformer, var])

    over_df.to_parquet('data/overperformer_imputed.parquet', index=True)
    under_df.to_parquet('data/underperformer_imputed.parquet', index=True)
    all_df.to_parquet('data/all_imputed.parquet', index=True)

    over_df_scaled.to_parquet('data/overperformer_sc.parquet', index=True)
    under_df_scaled.to_parquet('data/underperformer_sc.parquet', index=True)
    all_df_scaled.to_parquet('data/all_sc.parquet', index=True)


def save_data(data):
    #col_to_impute = get_numeric_col(data)
    #data[col_to_impute] = data[col_to_impute].fillna(data[col_to_impute].mean())

    overperformer, underperformer = get_outliers()
    all_df = data
    over_df = data.loc[overperformer, :]
    under_df = data.loc[underperformer, :]

    all_df.to_parquet('data/all.parquet', index=True)
    over_df.to_parquet('data/overperformer.parquet', index=True)
    under_df.to_parquet('data/underperformer.parquet', index=True)

def save_dashboard_variables():
    urban_form = ['Population', 'Area (2016)', 'Population Density', 'Built-up areas per capita'] 
    pct_metrics = ['Percentage Access to Public Transport', 'Share of population with access to open public spaces (2020)']
    city_iden = ['Country code', 'Region']
    urban_mobility = ['Road Length per capita (2017)', 'Pedestrian Streets per capita', 
                    'Modal split - Car (2017)', 'Modal split - Public Transport (2017)', 
                    'Rapid Transit to Resident Ratio (RTR)']
    
    try:
        with open('data/dashboard_var.json', 'w') as f:
            json.dump({'urban_form': urban_form,
                       'pct_metrics': pct_metrics,
                       'city_iden': city_iden,
                       'urban_mobility': urban_mobility}, f)
        return True
    except Exception as e:
        print("Error: ", e)
        return False
    
def get_dashboard_variables():
    with open("data/dashboard_var.json", "r") as f:
        var = json.load(f)

    return var['urban_form'], var['pct_metrics'], var['city_iden'], var['urban_mobility']
    
if __name__ == '__main__':
    # Train model
    model, std_data = run_pipeline(cleaned)
    processed = process_outputs(model, std_data)
    save_outputs(processed)

    # All variables, not normalised
    save_data(data)

    # Miscellaneous
    save_dashboard_variables()
    save_unit_dict()
    save_country_name_dict()

    # Normalised spider chart variables
    spider_chart_var = spider_chart_variables(data)
    save_spider_chart_variables(spider_chart_var)

    save_spider_chart_data(data)


