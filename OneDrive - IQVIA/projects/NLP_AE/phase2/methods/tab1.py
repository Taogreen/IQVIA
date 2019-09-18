import dash
import dash_html_components as html
import flask

from app import app

layout = html.Div([
	html.Img(src='/assets/bacgr.PNG',style={'width':'100%'})],style={"float":"center","height":"100%"})

