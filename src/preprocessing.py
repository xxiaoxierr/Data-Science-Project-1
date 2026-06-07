import pandas as pd
from sklearn.preprocessing import StandardScaler
import pwlf
import numpy as np
import json

# Load data (drop NaN)
def load_data(path):
    data = pd.read_csv(path)
    data.set_index('Urban Center', inplace=True)
    return data

def clean_data(data, returnall=False):
    X_col = ['Rapid Transit to Resident Ratio (RTR)', 'Oxford Economics Global Cities Index (Quality of Life)']
    y_col = 'Percentage Access to Public Transport'

    cleaned = data[X_col + [y_col]].copy()
    if ('London' in cleaned.index):
        cleaned = cleaned.drop('London')
    cleaned = cleaned.dropna()
    
    if returnall:
        cleaned = data.loc[cleaned.index, :]

    return cleaned

def rename_columns(data):
    return data.rename(columns={'country_code': 'Country code',
                         'region': 'Region', 
                         'Modal split (Passenger) - by trips - Car (2017)': 'Modal split - Car (2017)',
                         'Modal split (Passenger) - by trips - Public Transport (2017)': 'Modal split - Public Transport (2017)'})

# Standardise covariates only
def standardise(data):   
    X_col = ['Rapid Transit to Resident Ratio (RTR)', 'Oxford Economics Global Cities Index (Quality of Life)']
    y_col = 'Percentage Access to Public Transport' 
    X = data[X_col]
    y = data[y_col]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=X.columns
    )
    X_scaled[y_col] = y

    return X_scaled

# Create hinge variable for PWLF
def add_breakpoint(cleaned):    
    # Identify breakpoint by RTR
    pwlf_model = pwlf.PiecewiseLinFit(cleaned['Rapid Transit to Resident Ratio (RTR)'], 
                                      cleaned['Percentage Access to Public Transport'])
    breakpoints = pwlf_model.fit(2)
    knot_c = np.round(breakpoints[1],3)

    cleaned['rtr_high_hinge'] = np.maximum(0, cleaned['Rapid Transit to Resident Ratio (RTR)'] - knot_c)

    return cleaned


def prepare_features(cleaned):
    X_col = ['Rapid Transit to Resident Ratio (RTR)', 'Oxford Economics Global Cities Index (Quality of Life)', 'rtr_high_hinge']
    y_col = 'Percentage Access to Public Transport' 

    X = cleaned[X_col]
    y = cleaned[y_col]

    return X, y


def save_unit_dict():
    cleaned = pd.read_csv('data/cleaned.csv')
    raw = pd.read_csv('data/urban_data.csv')

    cleaned_indicators = cleaned.columns.tolist()

    # Create unique indicator-unit pair df
    result = []
    for indicator in cleaned_indicators:
        result.append(indicator.split('(2')[0].strip())
    unique_pairs = raw[raw['Indicator name'].isin(result)][['Indicator name', 'Units']]
    unique_pairs = unique_pairs.drop_duplicates().dropna()

    unit_dict = unique_pairs.set_index('Indicator name')['Units'].to_dict()

    # Save JSON
    with open("data/indicator_units.json", "w") as f:
        json.dump(unit_dict, f)

def save_country_name_dict():
    country_map = {'AFG': 'Afghanistan',
                'ARM': 'Armenia',
                'AUS': 'Australia',
                'AZE': 'Azerbaijan',
                'BGD': 'Bangladesh',
                'KHM': 'Cambodia',
                'CHN': 'China',
                'HKG': 'Hong Kong',
                'TWN': 'Taiwan',
                'GEO': 'Georgia',
                'IND': 'India',
                'IDN': 'Indonesia',
                'IRN': 'Iran',
                'JPN': 'Japan',
                'KGZ': 'Kyrgyzstan',
                'LAO': 'Laos',
                'MYS': 'Malaysia',
                'MNG': 'Mongolia',
                'MMR': 'Myanmar',
                'NPL': 'Nepal',
                'NZL': 'New Zealand',
                'PAK': 'Pakistan',
                'PHL': 'Philippines',
                'KOR': 'South Korea',
                'RUS': 'Russia',
                'SGP': 'Singapore',
                'THA': 'Thailand',
                'UZB': 'Uzbekistan',
                'VNM': 'Vietnam',
                'AGO': 'Angola',
                'ARG': 'Argentina',
                'BLR': 'Belarus',
                'BRA': 'Brazil',
                'CAN': 'Canada',
                'CHL': 'Chile',
                'COL': 'Colombia',
                'CIV': "Côte d'Ivoire",
                'COD': 'Democratic Republic of the Congo',
                'DOM': 'Dominican Republic',
                'ECU': 'Ecuador',
                'EGY': 'Egypt',
                'FRA': 'France',
                'DEU': 'Germany',
                'GRC': 'Greece',
                'HTI': 'Haiti',
                'IRQ': 'Iraq',
                'ITA': 'Italy',
                'JOR': 'Jordan',
                'KEN': 'Kenya',
                'MEX': 'Mexico',
                'NLD': 'Netherlands',
                'NGA': 'Nigeria',
                'PER': 'Peru',
                'PRT': 'Portugal',
                'SAU': 'Saudi Arabia',
                'ZAF': 'South Africa',
                'ESP': 'Spain',
                'SDN': 'Sudan',
                'TZA': 'Tanzania',
                'TUR': 'Turkey',
                'UKR': 'Ukraine',
                'ARE': 'United Arab Emirates',
                'GBR': 'United Kingdom',
                'USA': 'United States',
                'VEN': 'Venezuela',
                'BTN': 'Bhutan',
                'COK': 'Cook Islands',
                'KIR': 'Kiribati',
                'NIU': 'Niue',
                'WSM': 'Samoa',
                'TON': 'Tonga',
                'TUV': 'Tuvalu',
                'VUT': 'Vanuatu',
                'BRN': 'Brunei',
                'FJI': 'Fiji',
                'KAZ': 'Kazakhstan',
                'MDV': 'Maldives',
                'MHL': 'Marshall Islands',
                'FSM': 'Micronesia',
                'PLW': 'Palau',
                'PNG': 'Papua New Guinea',
                'SLB': 'Solomon Islands',
                'LKA': 'Sri Lanka',
                'TJK': 'Tajikistan',
                'TLS': 'Timor-Leste',
                'TKM': 'Turkmenistan',
                'GHA': 'Ghana'
            }
    
    # Save JSON
    with open("data/country_names.json", "w") as f:
        json.dump(country_map, f)
