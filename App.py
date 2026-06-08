import streamlit as st
import plotly.express as px
import altair as alt 

import pandas as pd
import numpy as np
import math
import re
import json

# 1. Load data
from src.preprocessing import load_data, clean_data
from src.run_pipeline import get_outliers, get_dashboard_variables
from pathlib import Path

def load_and_clean_data():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "data" / "all_cleaned.csv"

    df = load_data(DATA_PATH)
    df = clean_data(df, returnall=True)

    return df

# 3. Light preprocessing
def preprocessing(df):
    # Set ranking type to int
    df = df.astype({'Oxford Economics Global Cities Index (Rank)': 'Int64', 
                'Oxford Economics Global Cities Index (Economics)': 'Int64',
                'Oxford Economics Global Cities Index (Human Capital)': 'Int64',
                'Oxford Economics Global Cities Index (Quality of Life)': 'Int64',
                'Oxford Economics Global Cities Index (Environment)': 'Int64',
                'Oxford Economics Global Cities Index (Governance)': 'Int64'})

    # Rename columns
    df = df.rename(columns={'country_code': 'Country Code', 
                            'region': 'Region'})

    # Round float to 3 dp
    float_mask = (df.dtypes == 'float64')
    df.loc[:, float_mask] = df.loc[:, float_mask].round(3)

    return df

# 4. Interface
def display_ranking(city, df, compare=''):
    ranking_col = df.columns[df.columns.str.contains("Oxford Economics")].tolist()

    st.subheader('Oxford Economics Global Cities Index (2024)')

    # Extract text inside ()
    result = [] 
    for col in ranking_col: 
        matches = re.findall(r'\((.*?)\)', col)
        result.extend(matches)
    
    # Display ranking
    if compare == '':
        for i, name in enumerate(result):
            value = df.loc[city, ranking_col[i]]
            max_value = df[ranking_col[i]].max()

            # Bold text
            st.write(f"**{name}**")
            st.progress(float(value / max_value))  
            st.caption(f"{value} / {max_value}")
    else:
        bc_df = df.loc[[city, compare], ranking_col] 
        bc_df = bc_df.transpose()
        bc_df = bc_df.reset_index()

        # Rename indicators 
        bc_df = bc_df.rename(columns={'index': 'Domain'})
        bc_df['Domain'] = result
        df_melted = bc_df.melt(id_vars='Domain', var_name='City', value_name='Rank')
    
        bc_df_1 = bc_df.iloc[0:2, :]
        bc_df_2 = bc_df.iloc[2:4, :]
        bc_df_3 = bc_df.iloc[4:6, :]

        df_melted_1 = bc_df_1.melt(id_vars='Domain', var_name='City', value_name='Rank')
        df_melted_2 = bc_df_2.melt(id_vars='Domain', var_name='City', value_name='Rank')
        df_melted_3 = bc_df_3.melt(id_vars='Domain', var_name='City', value_name='Rank')

        to_plot = [df_melted_1, df_melted_2, df_melted_3]

        col1, col2 = st.columns([1,1])

        with col1:
            st.markdown(
                f"""
                <span style="color:#27847d">● </span> {city} 
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <span style="color:#1B5B70">● </span> {compare} 
                """,
                unsafe_allow_html=True
            )

        for plot in to_plot:
            fig = px.bar(
                plot, 
                x='Domain', 
                y='Rank', 
                color='City',
                barmode='group',      # side-by-side bars
                text_auto='.0f',       
                color_discrete_sequence=["#27847d", "#1B5B70"] 
            )

            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis=dict(zeroline=False),
                xaxis_title = None,
                yaxis_title = None,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)


# 1. Urban form
def display_urban_form(city, urban_form, df, compare=''):
    st.subheader("Urban Form")

    col1, col2 = st.columns([1,1])

    if compare == '':
        # Show integer for clarity
        for i, metric in enumerate(urban_form):
            value = df.loc[city, metric]

            if i <= 1:
                with col1:
                    fill_urban_form(metric, value)
            else:
                with col2:
                    fill_urban_form(metric, value)
    else:
        for i, metric in enumerate(urban_form):
            df_compare = df.loc[[city, compare], metric]
            value = df.loc[city, metric]
            delta = df.loc[city, metric] - df.loc[compare, metric]

            if i <= 1:
                with col1:
                    fill_urban_form(metric, value, delta)
            else:
                with col2:
                    fill_urban_form(metric, value, delta)


def fill_urban_form(metric, value, delta=''):
    if metric == 'Built-up areas per capita':
        metric_label = f"{metric}"
        if math.isnan(value):
            value_label = 'No data'
        else:
            value_label = str(int(value)) + " sqm"

    elif metric == 'Population':
        metric_label = metric
        if math.isnan(value):
            value_label = 'No data'
        else:
            value_label = format_number_M(value)

    else:
        metric_label = metric
        if math.isnan(value):
            value_label = 'No data'
        else:
            value_label = f"{int(value):,}"
    
    if delta == '':
        st.metric(label=metric_label, value=value_label)

    else:
        if metric == 'Built-up areas per capita':
            delta_label = int(delta)
        elif metric == 'Population':
            delta_label = format_number_M(delta)
        else:
            delta_label = f"{int(delta):,}"

        st.metric(label=metric_label, value=value_label, delta=delta_label)


def format_number_M(num):
    number = np.round(num / np.power(10,6), 1)
    return str(number) + " M"

def format_number_K(num):
    number = np.round(num / np.power(10,3), 1)
    return str(number) + " K"


def get_units(indicator):
    with open('data/indicator_units.json') as f:
        data = json.load(f)
    
    try:
        indicator = indicator.split('(2')[0].strip()
        return data[indicator]
    except Exception as e:
        print("Error: ", e)

# 2. Percentage metrics
def make_donut(percentage, indicator, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']
    
    # Convert to percentage 
    percentage *= 100

    # Color in front 
    source = pd.DataFrame({
        "Topic": ['', indicator],
        "% value": [100-percentage, percentage]
    })

    # Background
    source_bg = pd.DataFrame({
        "Topic": ['', indicator],
        "% value": [100, 0]
    })
    
    # Mark color in front
    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color= alt.Color("Topic:N",     # :N indicates nominal data
                         scale=alt.Scale(
                            domain=[indicator, ''],
                            range=chart_color),
                            legend=None),
        ).properties(width=130, height=130)
    
    # Text in the middle
    text = plot.mark_text(align='center', color="#29b5e8", fontSize=28, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{np.round(percentage, 1)} %'))

    # Mark background
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[indicator, ''],
                            range=chart_color),  
                            legend=None),
    ).properties(width=130, height=130)

    # Background, percentage arc, text
    return plot_bg + plot + text

def display_pct_metrics(city, pct_metrics, df, compare=''):
    colour = ['blue', 'green']
    
    if compare == '':
        for i, metric in enumerate(pct_metrics):
            value = df.loc[city, metric]
            chart = make_donut(value, metric, colour[i])

            st.markdown(f'<p style="font-size: 17px"> {metric} </p>', unsafe_allow_html=True)
            st.altair_chart(chart, use_container_width=True)

        st.markdown(f'<p style="font-size: 17px"> Data completness score </p>', unsafe_allow_html=True)
        display_data_completeness(city, df)

    else:
        for i, metric in enumerate(pct_metrics):
            value_1 = df.loc[city, metric]
            value_2 = df.loc[compare, metric]

            st.markdown(f'<p style="font-size: 17px"> {metric} </p>', unsafe_allow_html=True)

            col1, col2 = st.columns([1,1])
            chart_1 = make_donut(value_1, metric, colour[i])
            chart_2 = make_donut(value_2, metric, colour[i])

            with col1:
                st.text(city)
                st.altair_chart(chart_1, use_container_width=True)
            with col2:
                st.text(compare)
                st.altair_chart(chart_2, use_container_width=True)
        
        st.markdown(f'<p style="font-size: 17px"> Data completness score </p>', unsafe_allow_html=True)

        col1, col2 = st.columns([1,1])
        with col1:
            st.text(city)
            display_data_completeness(city, df)
        with col2:
            st.text(compare)
            display_data_completeness(compare, df)

# 3. City identity (country, region)
def display_city_iden(city, city_iden, df):
    st.title(city)
    for iden in city_iden:
        if iden == 'Country code':
            # Map country code to name
            code = df.loc[city, iden]
            with open('data/country_names.json', 'r') as f:
                country_map = json.load(f)
            st.subheader(f'{country_map[code]}')

        else:
            region = df.loc[city, iden]
            st.text(f"{region}")
    
# 4. Spider chart
def generate_spider_chart(city, df, df_imputed, compare=''):
    if compare == '':
        city_imputed_df = df_imputed.loc[city, :].reset_index()
        city_imputed_df.columns = ['indicator', 'value']

        city_df = df.loc[city, :].reset_index()
        city_df.columns = ['indicator', 'value']

        fig = px.line_polar(
            city_df,
            r = 'value',
            theta = 'indicator',
            line_close = True,
            template = 'plotly_dark'
        )

        fig.update_traces(
            fill='toself', 
            line_width=2,
            marker=dict(size=8, color="#204E84")
        )

        fig.update_layout(
            polar=dict(
                # Customise theta
                angularaxis=dict(
                    tickfont=dict(
                        size=16,          
                        weight="bold"     
                    ),
                    gridcolor="#E2E8F0"   
                ),

                # Customise r
                radialaxis=dict(
                    visible=True,
                    range=[-3,4]
                )
            ),
            showlegend=False,
            margin=dict(l=60, r=60, t=50, b=50),
            autosize = False
        )

    else:
        city_imputed_df = df_imputed.loc[[city,compare], :]
        city_df = df.loc[[city,compare], :]      

        long_df = (
            city_df.reset_index()
            .melt(
                id_vars='Urban Center',
                var_name='indicator',
                value_name='value'
            )
        )

        fig = px.line_polar(
            long_df,
            r = 'value',
            theta = 'indicator',
            color = 'Urban Center',
            line_close = True,
            template = 'plotly_dark'
        )

        fig.update_traces(
            fill='toself', 
            opacity=0.6,
            line_width=2,
            marker=dict(size=8, color="#204E84")
        )

        fig.update_layout(
        polar=dict(
            # Customise theta
            angularaxis=dict(
                tickfont=dict(
                    size=16,          
                    weight="bold"     
                ),
                gridcolor="#E2E8F0"   # Light grey grid rings
            ),

            # Customise r
            radialaxis=dict(
                visible=True,
                range=[-3,4]
            )
        ),
        showlegend=True,
        margin=dict(l=60, r=60, t=50, b=50),
        autosize = False
    )

    st.plotly_chart(fig, use_container_width=True)
    display_spider_chart_metric(city, df_imputed, compare)

def display_spider_chart_metric(city, df_imputed, compare=''):
    cols = st.columns(5)
    metrics = df_imputed.columns.tolist()

    if compare == '':
        for i, metric in enumerate(metrics):
            value = df_imputed.loc[city, metric]

            with cols[i]:
                if metric == 'Rapid Transit to Resident Ratio (RTR)':
                    metric_label = 'RTR'
                    value_label = str(np.round(value, 1)) + " km"

                elif metric == 'Modal split - Public Transport (2017)':
                    metric_label = 'Modal split - Public Transport'
                    value *= 100 
                    value_label = str(np.round(value, 1)) + ' %'

                elif metric == 'Modal split - Car (2017)':
                    metric_label = 'Modal split - Car'
                    value *= 100 
                    value_label = str(np.round(value, 1)) + ' %'

                elif metric == 'Pedestrian Streets per capita':
                    metric_label = metric
                    if value >= 1000:
                        value_label = format_number_K(np.round(value, 1)) + " m"
                    else:                     
                        value_label = str(np.round(value, 1)) + ' m'

                elif metric == 'Road Length per capita (2017)':
                    metric_label = metric
                    value_label = str(np.round(value, 1)) + ' m'
                
                st.metric(label=metric_label, value=value_label)
    else:
        for i, metric in enumerate(metrics):
            value1 = df_imputed.loc[city, metric]
            value2 = df_imputed.loc[compare, metric]

            delta_label = np.round(value1-value2, 1)

            with cols[i]:
                if metric == 'Rapid Transit to Resident Ratio (RTR)':
                    metric_label = 'RTR'
                    value_label = str(np.round(value1, 1)) + " km"

                elif metric == 'Modal split - Public Transport (2017)':
                    metric_label = 'Modal split - Public Transport'
                    value1 *= 100 
                    value2 *= 100
                    value_label = str(np.round(value1, 1)) + ' %'

                elif metric == 'Modal split - Car (2017)':
                    metric_label = 'Modal split - Car'
                    value1 *= 100 
                    value2 *= 100
                    value_label = str(np.round(value1, 1)) + ' %'

                elif metric == 'Pedestrian Streets per capita':
                    metric_label = metric
                    if value1 >= 1000:
                        value_label = format_number_K(value1) + " m"
                        delta_label = format_number_K(value1) + " m"
                    else:                     
                        value_label = str(np.round(value1, 1)) + ' m'

                elif metric == 'Road Length per capita (2017)':
                    metric_label = metric
                    value_label = str(np.round(value1, 1)) + ' m'
                
                st.metric(label=metric_label, value=value_label, delta=delta_label)
        


def display_data_completeness(city, df):
    score = 1- df.loc[city, :].isna().sum() / len(df.columns)

    chart = make_donut(np.round(score, 2), 'Data completeness score', 'orange')
    st.altair_chart(chart, use_container_width=True)

def display_indicator_units(indicators):
    st.subheader('Data glossary')

    with open("data/indicator_units.json", "r") as f:
        data = json.load(f)

    df = pd.DataFrame(columns=['Indicator', 'Units'])
    indicator_map = {'Modal split - Car (2017)': 'Modal split (Passenger) - by trips - Car (2017)', 
                     'Modal split - Public Transport (2017)': 'Modal split (Passenger) - by trips - Public Transport (2017)'}

    for indicator in indicators:
        if indicator in ['Modal split - Car (2017)', 'Modal split - Public Transport (2017)']:
            to_insert = indicator_map[indicator]
        else:
            to_insert = indicator

        to_insert = to_insert.split('(2')[0].strip()
        df.loc[len(df)] = [to_insert, data[to_insert]]

    st.dataframe(df, use_container_width=True, hide_index=True) 
        

def generate_dashboard(city, df, df_sc, df_imputed, compare=''):
    urban_form, pct_metrics, city_iden, urban_mobility = get_dashboard_variables()
    display_city_iden(city, city_iden, df)

    if compare == '':
        col1, col2 =  st.columns([1.75, 1])
        
        with col1:
            display_urban_form(city, urban_form, df)
            display_pct_metrics(city, pct_metrics, df)

        with col2: 
            display_ranking(city, df)
        
        generate_spider_chart(city, df_sc, df_imputed)
        display_indicator_units(urban_form+pct_metrics+urban_mobility)
    
    else:
        col1, col2 =  st.columns([1, 1])

        with col1:
            display_urban_form(city, urban_form, df, compare)
            st.space()
            display_pct_metrics(city, pct_metrics, df, compare)

        with col2:
            display_ranking(city, df, compare)


        generate_spider_chart(city, df_sc, df_imputed, compare)
        display_indicator_units(urban_form+pct_metrics+urban_mobility)


if __name__ == '__main__':
    df = pd.read_parquet('data/all.parquet')
    df = preprocessing(df)
    df_sc = pd.read_parquet('data/all_sc.parquet')
    df_imputed = pd.read_parquet('data/all_imputed.parquet')

    cities = df.index.tolist()
    over_cities, under_cities = get_outliers()

    tab1, tab2 = st.tabs(['Single view', 'Comparison view'], key='tab')

    with st.sidebar:
        type_choice = st.selectbox("What types of cities to explore?", ['All cities', 'Outliers'])

        if type_choice == 'All cities':
            city_choice = st.selectbox("I want to explore", cities)
            compare_cities = cities

        if type_choice == 'Outliers':
            outlier_choice = st.selectbox("Which outliers to explore?", ['Overperformer', 'Underperformer'])
            if outlier_choice == 'Overperformer':
                city_choice = st.selectbox("I want to explore", over_cities)
                compare_cities = over_cities

            if outlier_choice == 'Underperformer':
                city_choice = st.selectbox("I want to explore", under_cities)
                compare_cities = under_cities

    with tab1:
        generate_dashboard(city_choice, df, df_sc, df_imputed)
           
    with tab2:
        compare_choice = st.selectbox(f"I want to compare {city_choice} with", [x for x in compare_cities if x!=city_choice])
        if compare_choice:
            generate_dashboard(city_choice, df, df_sc, df_imputed, compare_choice)
        
        


    










    




