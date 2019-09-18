# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.abspath('.'))

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import flask
import tab1,tab2
from app import app


tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'font-size':'1.2em'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    #'backgroundColor': '#005487',
    'backgroundColor': '#00A1DF',
    'color': 'white',
    'padding': '6px',
    'fontWeight': 'bold',
    'font-size':'1.2em'
}



app.layout = html.Div(
 	[
        # header
        html.Div([

            #html.Span("KOL Scoring", className='app-title'),

            html.Div(
                html.Img(src='/assets/logo.PNG',height=60, width=260)
                ,style={"float":"right","height":"50%",}),
            html.Div([
                html.H1(children="AE Platform",style={'color': '#00A1DF','fontWeight': 'bold','fontFamily':'Arial','bottom-margin':'-5px'})
                # html.Img(src='/assets/sign1.PNG',height=65, width=420)
                # ,style={"height":"80%"}
                ],),
            ],
            #className="row header"
            ),

        # tabs
        html.Div([

            dcc.Tabs(
                id="tabs",
                #style={"height":"5","verticalAlign":"middle"},
                children=[
                    #dcc.Tab(id = 'Scoring',label="评分", value="tab1",style={'padding': '0','font-size': '1.2em','fontWeight': 'bold','fontSize':'14','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}),
                    dcc.Tab(id ='Instruction',label="Welcome", value="tab1",style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(id="Detector",label="AE Detector", value="tab2",style=tab_style, selected_style=tab_selected_style),
                    
                ],style=tabs_styles,
                value = "tabs"

            )

            ],
            #className="row tabs_div"
            ),
        # Tab content
        html.Div(id="tab_content", className="row", style={"margin": "2% 3%"}),

        html.Link(href="https://use.fontawesome.com/releases/v5.2.0/css/all.css",rel="stylesheet"),
        html.Link(href="https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css",rel="stylesheet"),
        html.Link(href="https://fonts.googleapis.com/css?family=Dosis", rel="stylesheet"),
        html.Link(href="https://fonts.googleapis.com/css?family=Open+Sans", rel="stylesheet"),
        html.Link(href="https://fonts.googleapis.com/css?family=Ubuntu", rel="stylesheet"),
        #html.Link(href="https://cdn.rawgit.com/amadoukane96/8a8cfdac5d2cecad866952c52a70a50e/raw/cd5a9bf0b30856f4fc7e3812162c74bfc0ebe011/dash_crm.css", rel="stylesheet")
	],
	#className="row",
	style={"margin":"0%"},
)




@app.callback(Output("tab_content", "children"), [Input("tabs", "value")])
def render_content(tab):
    if tab == "tab1":
        return tab1.layout
    elif tab == "tab2":
        return tab2.layout
    
    else:
        return tab1.layout



if __name__ == "__main__":
    app.run_server(debug=True)
