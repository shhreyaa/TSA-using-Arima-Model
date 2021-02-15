import pandas as pd
import dash
import dash_table
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import datetime
import flask
import plotly.express as px 
import plotly.graph_objs as go # (need to pip install plotly==4.4.1)
import plotly
# import warnings
# import itertools
# import numpy as np
# import matplotlib.pyplot as plt
# import chart_studio
# import statsmodels.api as sm
# import matplotlib
# import chart_studio.plotly as py
# from chart_studio.plotly import plot, iplot






# from pmdarima import auto_arima
# chart_studio.tools.set_credentials_file(username='shhreyaa',                                              
#                                   api_key='UGJfLKcZ6nznR80PcPL8')

app = dash.Dash(__name__)

# ---------- Import and clean data (importing csv into pandas)


# df = pd.read_excel("sales_11.xlsx")
df = pd.read_excel("sales_replaced.xlsx")
df['Order Date'] = pd.to_datetime(df['Order Date'])
future_forcast = pd.read_excel("future_work.xlsx")
future_forcast['Date'] = pd.to_datetime(future_forcast['Date']).dt.date
furniture_forcast=pd.read_excel("furniture_work.xlsx")
office_forcast=pd.read_excel("office_work.xlsx")
technology_forcast=pd.read_excel("technology_work.xlsx")


#-----------------------------------------------------
#Total sale prediction 
sale_total = df.copy()

sale_total = pd.read_excel("sales_replaced.xlsx")
sale_total.head()
cols = ['Row ID', 'Customer ID', 'Segment', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
sale_total.drop(cols, axis=1, inplace=True)
sale_total = sale_total.sort_values('Order Date')


sale_total.isnull().sum()
sale_total= sale_total.groupby('Order Date')['Sales'].sum().reset_index()
sale_total = sale_total.set_index('Order Date')

sale_total=sale_total['Sales'].resample('MS').sum()

sale_total= sale_total.to_frame()





# train = sale_total[:'2017-01-01']
# train.head()
# test= sale_total['2017-01-01':]
# plt.plot(train)
# plt.plot(test)
# print(sale_total.index)
# print(sale_total['Sales'])
# # print(sale_total['Order Date'])



# fig = px.line(future_forcast, x="Date", y="Prediction")
# fig= fig.add_scatter(x=sale_total["Order Date"], y=sale_total["Total Sale"], mode='lines')
fig = go.Figure([
      go.Scatter(
        name='Predicted value',
        x=future_forcast['Date'],
        y=future_forcast['Prediction'],
        mode='markers+lines',
        marker=dict(color='red', size=5),
        showlegend=True
    ),
     go.Scatter(
        name='Actual sales',
        x=sale_total.index,
        y=sale_total['Sales'],
        mode='markers+lines',
        marker=dict(color='blue', size=5),
        showlegend=True
    )

])

fig.update_layout(
   
   height=700,
   
  
    margin=dict(
        l=50,
        r=50,
        b=10,
        t=50,
        pad=2
    ),
    # paper_bgcolor="LightSteelBlue",
    # plot_bgcolor="LightSteelBlue",
)



# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([
    html.Div([
         html.H1("Web Application Dashboards with Dash", style={'text-align': 'center'})
         ]),
    html.Div([

   
    dcc.Graph(id='my_line', figure=fig)

],
 style={'width': '70%', 'display': 'inline-block',
               'backgroundColor': '#22C1AD'
               }),
    html.Div([
         html.Div([
         html.H1("17.6%", style={'text-align': 'center'}),
         html.H2("Estimated growth for the upcoming year")
         ],style={'width': '70%', 'display': 'inline-block', 'padding':'5%'
               }
                  ),
           html.Br(),
             html.Br(),
               html.Br(),
    dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in future_forcast.columns],
  
    data=future_forcast.to_dict('records'),
    style_table={'height': '400px', 'overflowY': 'auto'},
      fixed_rows={'headers': True},
      style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }
    ],
        style_cell_conditional=[
        {
           
            'textAlign': 'left'
        }],
    style_header={
        'backgroundColor': 'rgb(230, 230, 230)',
        'fontWeight': 'bold'
    },
    
)
],
              style={'width': '29%', 'display': 'inline-block',
                     'align':'right','position':'absolute','margin':'50px 0px'
               }),
    

#side component
#For individual prediction
html.Div([
   
    dcc.RadioItems(
    options=[
        {'label': 'Furniture', 'value': 'Furniture'},
        {'label': 'Office Supplies', 'value': 'Office Supplies'},
        {'label': 'Technology', 'value': 'Technology'}
    ],
    id= 'radio-items',
    value='Technology',
    labelStyle={'display': 'inline-block'}
),
    html.Div(id='output_container', children=[]),
    html.Br(),
    
    dcc.Graph(
    id='individual_pred'),
    html.H1(
    id='expected_growth', children=[]),
    
]),

])





# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback([Output('individual_pred', 'figure'),
               Output('output_container', 'children'),
               Output('expected_growth', 'children')
               ],
              
    [Input('radio-items', 'value')])


def make_line_chart(value):
    container = "Time Series Prediction Graph for {} for 2018-2020 ".format(value)
    sale_category = df.loc[df["Category"] == "{}".format(value)]
    cols = ['Row ID', 'Customer ID', 'Segment', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
    sale_category.drop(cols, axis=1, inplace=True)
    sale_category = sale_category.sort_values('Order Date')

    sale_category.isnull().sum()
    sale_category= sale_category.groupby('Order Date')['Sales'].sum().reset_index()
    sale_category = sale_category.set_index('Order Date')
    sale_category=sale_category['Sales'].resample('MS').sum()
    sale_category= sale_category.to_frame()
    if (value == "Furniture"):
        dff = furniture_forcast
        expected_growth = "Expected Growth for {} is 9.87% ".format(value)
        
    elif (value =="Office Supplies"):
        dff =office_forcast
        expected_growth = "Expected Growth for {} is 10.46% ".format(value)
    else:
        dff= technology_forcast
        expected_growth = "Expected Growth for {} is 16.45% ".format(value)
        
    
    
    
    figure = go.Figure([
      go.Scatter(
        name='Predicted value',
        x=dff['Date'],
        y=dff['Prediction'],
        mode='markers+lines',
        marker=dict(color='red', size=5),
        showlegend=True
    ),
     go.Scatter(
        name='Actual sales',
        x=sale_category.index,
        y=sale_category['Sales'],
        mode='markers+lines',
        marker=dict(color='blue', size=5),
        showlegend=True
    )


    ])
    return figure,container,expected_growth

    
  





# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)