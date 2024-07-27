import base64
import io

import dash
import numpy as np
from dash import dcc, html
from dash.dash_table import DataTable
from dash.dependencies import Input, Output, State
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from EDA import *
import seaborn as sns
import matplotlib.pyplot as plt


# Assume df_train is your DataFrame
# Example DataFrame creation:
df_train = pd.read_csv('train.csv')
years_options = [{'label': str(year), 'value': year} for year in range(2006, 2010)]
# Update the model names list without Linear Regression
model_names = ['Random Forest', 'Support Vector', 'Gradient Boosting', 'XGBoost']

# Create the DataFrame for results
results_df = pd.DataFrame({
    'Model': model_names,
    'Test RMSE': [
        rmse_test_rf, rmse_test_svr, rmse_test_gb, rmse_test_xgb
    ]
})

# Create the bar chart using Plotly Express
fig = px.bar(results_df, x='Model', y='Test RMSE', title='Model Test RMSE Comparison')

model_names_New = ['Random Forest', 'Support Vector', 'Gradient Boosting', 'Linear Regression', 'Decision Tree', 'XGBoost']
r_squared_train = [rf_model.score(X_train, y_train),
                   svr_model.score(X_train, y_train),
                   gb_model.score(X_train, y_train),
                   lr_model.score(X_train, y_train),
                   dt_model.score(X_train, y_train),
                   xgb_model.score(X_train, y_train)]
r_squared_test = [rf_model.score(X_test, y_test),
                  svr_model.score(X_test, y_test),
                  gb_model.score(X_test, y_test),
                  lr_model.score(X_test, y_test),
                  dt_model.score(X_test, y_test),
                  xgb_model.score(X_test, y_test)]

# Create a DataFrame for R-squared scores
results_df_New = pd.DataFrame({
    'Model': model_names_New,
    'Training R-squared': r_squared_train,
    'Test R-squared': r_squared_test
})

# Update the layout of the chart
fig.update_layout(
    xaxis_title='Model',
    yaxis_title='Test RMSE',
    yaxis=dict(tickformat='.2f'),  # Format y-axis ticks to two decimal places
)

def create_bedroom_histogram():
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_train['BedroomAbvGr'], name='BedroomAbvGr'))
    fig.update_layout(title='Total Number Of Bedroom Histogram', xaxis_title='Bedrooms', yaxis_title='Count')
    return fig

# Define function to create pie chart for TotRmsAbvGrd
def create_rooms_pie_chart():
    rooms_counts = df_train['TotRmsAbvGrd'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=rooms_counts.index, values=rooms_counts.values)])
    fig.update_layout(title='Total Number Of Rooms Pie Chart')
    return fig
# Create the box plots
# Create actual vs predicted price graph for each model
def create_model_graph(model, name):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    fig = go.Figure()

    # Add actual line for training set
    fig.add_trace(go.Scatter(
        x=y_train.index,
        y=y_train,
        mode='lines',
        name=f'Actual {name} (Training)',
        line=dict(color='blue'),
    ))

    # Add predicted line for training set
    fig.add_trace(go.Scatter(
        x=y_train.index,
        y=y_pred_train,
        mode='lines',
        name=f'Predicted {name} (Training)',
        line=dict(color='red'),
    ))

    # Add actual line for testing set
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test,
        mode='lines',
        name=f'Actual {name} (Testing)',
        line=dict(color='green'),
    ))

    # Add predicted line for testing set
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_pred_test,
        mode='lines',
        name=f'Predicted {name} (Testing)',
        line=dict(color='orange'),
    ))

    fig.update_layout(
        title=f'Actual vs Predicted Prices - {name}',
        xaxis_title='Index',
        yaxis_title='Prices',
        showlegend=True,
        legend=dict(
            x=0.1,
            y=0.9,
            traceorder='normal'
        )
    )

    return fig
def create_model_graph(model_names, train_rmse, test_rmse):
    fig = go.Figure()

    # Add line for training RMSE
    fig.add_trace(go.Scatter(
        x=model_names,
        y=train_rmse,
        mode='lines+markers',
        name='Training RMSE',
        marker=dict(color='blue'),
    ))

    # Add line for testing RMSE
    fig.add_trace(go.Scatter(
        x=model_names,
        y=test_rmse,
        mode='lines+markers',
        name='Testing RMSE',
        marker=dict(color='red'),
    ))

    fig.update_layout(
        title='Training vs Testing RMSE Comparison',
        xaxis_title='Model',
        yaxis_title='RMSE',
        showlegend=True,
        legend=dict(
            x=0.1,
            y=0.9,
            traceorder='normal'
        )
    )

    return fig



def create_box_plots():
    fig = go.Figure()

    # Box plot for OverallQual vs SalePrice
    fig.add_trace(go.Box(
        y=df_train['SalePrice'],
        x=df_train['OverallQual'],
        name='OverallQual vs SalePrice'
    ))

    # Box plot for OverallCond vs SalePrice
    fig.add_trace(go.Box(
        y=df_train['SalePrice'],
        x=df_train['OverallCond'],
        name='OverallCond vs SalePrice'
    ))

    fig.update_layout(
        title='Relationship with categorical features',
        xaxis=dict(
            title='Categorical Features',
            showgrid=False
        ),
        yaxis=dict(
            title='SalePrice',
            showgrid=False
        ),
        boxmode='group'
    )

    return fig


def create_corr_heatmap():
    # Exclude non-numeric columns
    numeric_columns = df_train.select_dtypes(include=['number']).columns
    corrmat = df_train[numeric_columns].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corrmat.values,
        x=corrmat.columns,
        y=corrmat.columns,
        colorscale='Viridis',
        zmin=-1,
        zmax=1
    ))

    fig.update_layout(
        title='Correlation Matrix',
        xaxis=dict(
            title='Features'
        ),
        yaxis=dict(
            title='Features'
        )
    )

    return fig

def create_xgb_graph(model, X_train, X_test, y_train, y_test):
    fig = go.Figure()

    # Add actual vs predicted points for training set
    fig.add_trace(go.Scatter(
        x=y_train,
        y=model.predict(X_train),
        mode='markers',
        name='XGBoost (Training)',
        marker=dict(color='blue'),
        text=f'Training RMSE: {rmse_train_xgb:.2f}',
    ))

    # Add actual vs predicted points for testing set
    fig.add_trace(go.Scatter(
        x=y_test,
        y=model.predict(X_test),
        mode='markers',
        name='XGBoost (Testing)',
        marker=dict(color='red'),
        text=f'Testing RMSE: {rmse_test_xgb:.2f}',
    ))

    fig.add_trace(go.Scatter(
        x=[min(y_train), max(y_train)],
        y=[min(y_train), max(y_train)],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Ideal Line'
    ))

    fig.update_layout(
        title='Actual vs Predicted Prices - XGBoost',
        xaxis_title='Actual Prices',
        yaxis_title='Predicted Prices',
        showlegend=True,
        legend=dict(
            x=0.1,
            y=0.9,
            traceorder='normal'
        )
    )

    return fig

# Create the heatmap of the correlation matrix for top variables correlated with 'SalePrice'
def create_saleprice_corr_heatmap():
    # Filter out non-numeric columns
    numeric_columns = df_train.select_dtypes(include=['number']).columns
    df_numeric = df_train[numeric_columns]

    # Calculate correlation matrix
    k = 10  # Number of variables for heatmap
    corrmat = df_numeric.corr()
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_numeric[cols].values.T)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=cols,
        y=cols,
        colorscale='Viridis',
        zmin=-1,
        zmax=1
    ))

    fig.update_layout(
        title='Top Variables Correlated with SalePrice',
        xaxis=dict(
            title='Features'
        ),
        yaxis=dict(
            title='Features'
        )
    )

    return fig
navbar = dbc.Navbar(
    [
        html.Img(src="/assets/vector-house-home-buildings-logo-icons-template-removebg-preview.png", height="50px", className="ml-auto"),
        dbc.NavbarBrand("House Price Prediction", className="ml-2"),
    ],
    style={'justify-content':'center','pading':'10px'},
    color="primary",
    dark=False,
)
# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(

    children=[
    navbar,

    dbc.Row([
        dbc.Col(
            dbc.Card(
            DataTable(
            id='description-table',
            columns=[{'name': col, 'id': col} for col in df_train.describe().columns],
            data=df_train.describe().to_dict('records'),
            style_table={'overflowX': 'auto'},  # Enable horizontal scrolling
            style_cell={'minWidth': 95, 'maxWidth': 200, 'width': 200},  # Adjust cell widths
            fixed_rows={'headers': True},  # Fix headers while scrolling
        ),
                body=True,
                className='card-border',  # Custom CSS class for card border
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                html.Div(
                    id='columns-text',
                    children=[html.Span(col, style={'margin-right': '5px', 'font-family': 'Arial'}) for col in df_train.columns],
                    style={'display': 'flex', 'flex-wrap': 'wrap', 'align-items': 'center', 'width': '100%'}
                ),
                body=True,
                className='card-border',  # Custom CSS class for card border
            ),
            width=6
        )
    ], className='mb-3'),  # Add margin-bottom to the row

    dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(
                    id='saleprice-distplot',
                    figure=go.Figure(px.histogram(df_train, x='SalePrice', nbins=20, title='SalePrice Distribution'))
                ),
                body=True,
                className='card-border',  # Custom CSS class for card border
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                html.Div(id='distribution-characteristics'),
                body=True,
                className='card-border',  # Custom CSS class for card border
            ),
            width=6
        )
    ], className='mb-3'),  # Add margin-bottom to the row
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='bedroom-histogram', figure=create_bedroom_histogram()),
                body=True,
                className='card-border',
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='rooms-pie-chart', figure=create_rooms_pie_chart()),
                body=True,
                className='card-border',
            ),
            width=6
        )
    ], className='mb-3'),

    dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='scatter-plots'),
                body=True,
                className='card-border',  # Custom CSS class for card border
            ),
            width=12
        )
    ], className='mb-3'),  # Add margin-bottom to the row

    dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='box-plots', figure=create_box_plots()),
                body=True,
                className='card-border',  # Custom CSS class for card border
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='corr-heatmap', figure=create_corr_heatmap()),
                body=True,
                className='card-border',  # Custom CSS class for card border
            ),
            width=6
        )
    ], className='mb-3'),  # Add margin-bottom to the row

    dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='saleprice-corr-heatmap', figure=create_saleprice_corr_heatmap()),
                body=True,
                className='card-border',  # Custom CSS class for card border
            ),
            width=6
        ),
dbc.Col(
            dbc.Card(
                [
                    dcc.Graph(id='price-graph'),
                    dcc.Dropdown(
                        id='year-dropdown',
                        options=years_options,
                        value=2009,
                        placeholder='Select a Year',
                    ),
                ],
                body=True,
                className='card-border',  # Custom CSS class for card border
            ),
            width=6
        )
    ], className='mb-3'),  # Add margin-bottom to the row

    dbc.Row([
            dbc.Col(
                dbc.Card(
                    dcc.Graph(
                        id='actual-vs-predicted-svr',  # Update ID to reflect SVR model
                        figure={
                            'data': [
                                go.Scatter(
                                    x=y_train,
                                    y=y_pred_train_svr,  # Assuming y_pred_train_svr contains predictions for SVR
                                    mode='markers',
                                    name='Training Data',
                                    marker=dict(color='blue'),
                                    text=f'Training RMSE: {rmse_train_svr:.2f}',
                                ),
                                go.Scatter(
                                    x=y_test,
                                    y=y_pred_test_svr,  # Assuming y_pred_test_svr contains predictions for SVR
                                    mode='markers',
                                    name='Testing Data',
                                    marker=dict(color='red'),
                                    text=f'Testing RMSE: {rmse_test_svr:.2f}',
                                ),
                                go.Scatter(
                                    x=[min(y_train), max(y_train)],
                                    y=[min(y_train), max(y_train)],
                                    mode='lines',
                                    name='Ideal Line',
                                    line=dict(color='black', dash='dash'),
                                ),
                            ],
                            'layout': {
                                'title': 'Actual vs Predicted Prices - Support Vector Regressor',  # Update title accordingly
                                'xaxis': {'title': 'Actual Prices'},
                                'yaxis': {'title': 'Predicted Prices'},
                                'showlegend': True,
                                'legend': {'x': 0.1, 'y': 0.9, 'traceorder': 'normal'},
                            }
                        }
                    ),
                ),
                width=6
            ),
                    dbc.Col(
            dbc.Card(
                dcc.Graph(
                    id='actual-vs-predicted-gb',
                    figure={
                        'data': [
                            go.Scatter(
                                x=y_train,
                                y=y_pred_train_gb,
                                mode='markers',
                                name='Training Data',
                                marker=dict(color='blue'),
                                text=f'Training RMSE: {rmse_train_gb:.2f}',
                            ),
                            go.Scatter(
                                x=y_test,
                                y=y_pred_test_gb,
                                mode='markers',
                                name='Testing Data',
                                marker=dict(color='red'),
                                text=f'Testing RMSE: {rmse_test_gb:.2f}',
                            ),
                            go.Scatter(
                                x=[min(y_train), max(y_train)],
                                y=[min(y_train), max(y_train)],
                                mode='lines',
                                name='Ideal Line',
                                line=dict(color='black', dash='dash'),
                            ),
                        ],
                        'layout': {
                            'title': 'Actual vs Predicted Prices - Gradient Boosting',
                            'xaxis': {'title': 'Actual Prices'},
                            'yaxis': {'title': 'Predicted Prices'},
                            'showlegend': True,
                            'legend': {'x': 0.1, 'y': 0.9, 'traceorder': 'normal'},
                        }
                    }
                ),
            ),
            width=6
        )
    ], className='mb-3'),  # Add margin-bottom to the row

    dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(
                    id='actual-vs-predicted-svr',
                    figure={
                        'data': [
                            go.Scatter(
                                x=y_train,
                                y=y_pred_train_svr,
                                mode='markers',
                                name='Training Data',
                                marker=dict(color='blue'),
                                text=f'Training RMSE: {rmse_train_svr:.2f}',
                            ),
                            go.Scatter(
                                x=y_test,
                                y=y_pred_test_svr,
                                mode='markers',
                                name='Testing Data',
                                marker=dict(color='red'),
                                text=f'Testing RMSE: {rmse_test_svr:.2f}',
                            ),
                            go.Scatter(
                                x=[min(y_train), max(y_train)],
                                y=[min(y_train), max(y_train)],
                                mode='lines',
                                name='Ideal Line',
                                line=dict(color='black', dash='dash'),
                            ),
                        ],
                        'layout': {
                            'title': 'Actual vs Predicted Prices - Support Vector Regressor',
                            'xaxis': {'title': 'Actual Prices'},
                            'yaxis': {'title': 'Predicted Prices'},
                            'showlegend': True,
                            'legend': {'x': 0.1, 'y': 0.9, 'traceorder': 'normal'},
                        }
                    }
                ),
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                dcc.Graph(
                    id='actual-vs-predicted-lr',
                    figure={
                        'data': [
                            go.Scatter(
                                x=y_train,
                                y=y_pred_train_lr,
                                mode='markers',
                                name='Training Data',
                                marker=dict(color='blue'),
                                text=f'Training RMSE: {rmse_train_lr:.2f}',
                            ),
                            go.Scatter(
                                x=y_test,
                                y=y_pred_test_lr,
                                mode='markers',
                                name='Testing Data',
                                marker=dict(color='red'),
                                text=f'Testing RMSE: {rmse_test_lr:.2f}',
                            ),
                            go.Scatter(
                                x=[min(y_train), max(y_train)],
                                y=[min(y_train), max(y_train)],
                                mode='lines',
                                name='Ideal Line',
                                line=dict(color='black', dash='dash'),
                            ),
                        ],
                        'layout': {
                            'title': 'Actual vs Predicted Prices - Linear Regression',
                            'xaxis': {'title': 'Actual Prices'},
                            'yaxis': {'title': 'Predicted Prices'},
                            'showlegend': True,
                            'legend': {'x': 0.1, 'y': 0.9, 'traceorder': 'normal'},
                        }
                    }
                ),
            ),
            width=6
        )
    ], className='mb-3'),  # Add margin-bottom to the row

    dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(
                    id='xgb-graph',
                    figure=create_xgb_graph(xgb_model, X_train, X_test, y_train, y_test)
                ),
                body=True,
                className='card-border',  # Custom CSS class for card border
            ),
            width=12
        ),
        dbc.Col(
            dbc.Card(
                dcc.Graph(figure=fig),
                body=True,
                className='card-border',  # Custom CSS class for card border
            ),
            width=12
        ),
    ], className='mb-3'),  # Add margin-bottom to the row
    dbc.Row([
        html.H1(
            children="Predictor",
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        html.Label(
            children="Garage Area",
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        dcc.Input(
            id='input-garagearea',
            type='number',
            pattern="[0-9]*",
            placeholder='mean will be considered',
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        html.Label(
            children="No Of Bathrooms",
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        dcc.Input(
            id='input-fullbath',
            type='number',
            pattern="[0-9]*",
            placeholder='mean will be considered',
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        html.Label(
            children="Year Of Built",
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        dcc.Input(
            id='input-yearbuilt',
            type='number',
            pattern="[0-9]*",
            placeholder='mean will be considered',
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        html.Label(
            children="Lot Area",
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        dcc.Input(
            id='input-lotarea',
            type='number',
            pattern="[0-9]*",
            placeholder='mean will be considered',
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        html.Label(
            children="Total Basement Area",
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        dcc.Input(
            id='input-totalbsmtsf',
            type='number',
            pattern="[0-9]*",
            placeholder='mean will be considered',
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        html.Label(
            children="Total Living Area",
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        dcc.Input(
            id='input-grlivarea',
            type='number',
            pattern="[0-9]*",
            placeholder='mean will be considered',
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        html.Label(
            children="Total Number Of Bedrooms",
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        dcc.Input(
            id='input-bedroomabvgr',
            type='number',
            pattern="[0-9]*",
            placeholder='mean will be considered',
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        html.Label(
            children="Garage Cars",
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        dcc.Input(
            id='input-garagecars',
            type='number',
            pattern="[0-9]*",
            placeholder='mean will be considered',
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        html.Label(
            children="Total Rooms",
            style={'margin-bottom': '5px', 'display': 'block','text-align': 'center'}
        ),
        dcc.Input(
            id='input-totrmsabvgrd',
            type='number',
            pattern="[0-9]*",
            placeholder='mean will be considered',
            style={'margin-bottom': '10px', 'border-radius': '20px', 'text-align': 'center'}
            # Round input field corners and center text
        ),
        html.Button(
            'Predict',
            id='predict-button',
            n_clicks=0,
            style={'margin-bottom': '20px', 'border-radius': '20px',
                   'box-shadow': '0px 0px 15px 5px rgba(255, 255, 255, 0.75)'}
            # Round button corners and add glowing effect
        ),
        html.Div(id='output-prediction', style={'font-weight': 'bold'}),
    ] ,   style={
        'font-family': 'Arial, sans-serif',
        'max-width': '500px',
        'margin': 'auto',
        'padding': '20px',
        'background-color': '#f9f9f9',
        'border-radius': '10px',
        'box-shadow': '0 4px 8px rgba(0,0,0,0.1)',
    },
    )

])


# Calculate distribution characteristics
def calculate_characteristics():
    skewness = df_train['SalePrice'].skew()
    kurtosis = df_train['SalePrice'].kurtosis()
    characteristics_text = f"Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}"
    return characteristics_text

# Callback to update the distribution characteristics text
@app.callback(
    Output('distribution-characteristics', 'children'),
    Input('saleprice-distplot', 'figure')
)
def update_characteristics(figure):
    characteristics_text = calculate_characteristics()
    return html.Div(characteristics_text)


def create_scatter_plots():
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Living Area vs SalePrice', 'Total Basement Area vs SalePrice'))

    # Scatter plot for GrLivArea vs SalePrice
    trace1 = go.Scatter(
        x=df_train['GrLivArea'],
        y=df_train['SalePrice'],
        mode='markers',
        marker=dict(color='blue'),
        name='GrLivArea vs SalePrice'
    )
    fig.add_trace(trace1, row=1, col=1)

    # Scatter plot for TotalBsmtSF vs SalePrice
    trace2 = go.Scatter(
        x=df_train['TotalBsmtSF'],
        y=df_train['SalePrice'],
        mode='markers',
        marker=dict(color='red'),
        name='TotalBsmtSF vs SalePrice'
    )
    fig.add_trace(trace2, row=1, col=2)

    fig.update_xaxes(title_text='Living Area', row=1, col=1)
    fig.update_yaxes(title_text='SalePrice', row=1, col=1)
    fig.update_xaxes(title_text='Total Basement Area', row=1, col=2)
    fig.update_yaxes(title_text='SalePrice', row=1, col=2)

    fig.update_layout(showlegend=False)

    return fig


# Callback to update the scatter plots
@app.callback(
    Output('scatter-plots', 'figure'),
    Input('scatter-plots', 'id')
)
def update_plots(value):
    fig = create_scatter_plots()
    return fig
# Create the box plots
def create_box_plots():
    fig = go.Figure()

    # Box plot for OverallQual vs SalePrice
    fig.add_trace(go.Box(
        y=df_train['SalePrice'],
        x=df_train['OverallQual'],
        name='Overall Quality vs SalePrice'
    ))

    # Box plot for OverallCond vs SalePrice
    fig.add_trace(go.Box(
        y=df_train['SalePrice'],
        x=df_train['OverallCond'],
        name='Overall Condititon  vs SalePrice'
    ))

    fig.update_layout(
        title='Relationship with categorical features',
        xaxis=dict(
            title='Categorical Features',
            showgrid=False
        ),
        yaxis=dict(
            title='SalePrice',
            showgrid=False
        ),
        boxmode='group'
    )

    return fig


@app.callback(
    Output('price-graph', 'figure'),
    [Input('year-dropdown', 'value')]
)
def update_graph(selected_year):
    # Filter the dataset based on the selected year
    filtered_df = df_train[df_train['YrSold'] == selected_year]

    # Calculate the average price for each neighborhood in the filtered dataset
    avg_prices = filtered_df.groupby('Neighborhood')['SalePrice'].mean().reset_index()

    # Create the bar chart using Plotly Express
    fig = px.bar(avg_prices, x='Neighborhood', y='SalePrice',
                 title=f'Average Price by Neighborhood for Year {selected_year}',
                 labels={'SalePrice': 'Average Price', 'Neighborhood': 'Neighborhood'})

    return fig
# Define callback to update the graph based on user input (if needed)
df = pd.read_csv('train.csv')  # Replace 'train.csv' with the actual dataset file name

# Drop categorical features for training
numerical_columns_pr = df.select_dtypes(include=['int64', 'float64']).drop(columns=['SalePrice'])

# Calculate the mean values for numerical columns
mean_values_pr = numerical_columns_pr.mean()

# Fill missing values with mean
filled_numerical_columns_pr = numerical_columns_pr.fillna(mean_values_pr)

# Initialize the scaler and fit it to the data
scaler = StandardScaler()
scaled_numerical_columns = scaler.fit_transform(filled_numerical_columns_pr)

# Train the XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(scaled_numerical_columns, df['SalePrice'])
@app.callback(
    Output('output-prediction', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('input-garagearea', 'value'),
     State('input-fullbath', 'value'),
     State('input-yearbuilt', 'value'),
     State('input-lotarea', 'value'),
     State('input-totalbsmtsf', 'value'),
     State('input-grlivarea', 'value'),
     State('input-bedroomabvgr', 'value'),
     State('input-garagecars', 'value'),
     State('input-totrmsabvgrd', 'value')
     ]
)
def predict_price(n_clicks, garage_area, full_bath, year_built, lot_area, total_bsmt_sf, grliv_area, bedroom_abv_gr  , garage_cars, totrms_abv_grd):
    if n_clicks > 0:
        # Prepare input data
        input_data = {
            'GarageArea': [garage_area],
            'FullBath': [full_bath],
            'YearBuilt': [year_built],
            'LotArea': [lot_area],
            'TotalBsmtSF': [total_bsmt_sf],
            'GrLivArea': [grliv_area],
            'BedroomAbvGr': [bedroom_abv_gr],
            'GarageCars': [garage_cars],
            'TotRmsAbvGrd': [totrms_abv_grd]
        }

        # Create DataFrame with same columns as the training data
        input_df = pd.DataFrame(input_data, columns=numerical_columns_pr.columns)

        # Fill missing values with mean
        filled_input_df = input_df.fillna(mean_values_pr)

        # Scale input data using the same scaler used for training
        scaled_input_df = scaler.transform(filled_input_df)

        # Use the XGBoost model for prediction
        prediction = xgb_model.predict(scaled_input_df)

        return f"The predicted price is ${prediction[0]:,.2f}"
    else:
        return ""



# Run the appx
if __name__ == '__main__':
    app.run_server(debug=True)
