import flask
from flask import Flask
import dash
import dash_bootstrap_components as dbc

app = dash.Dash()
server = Flask(__name__)

app.config.suppress_callback_exceptions = True
app.scripts.config.serve_locally=True
app.title = 'AE_Platform'



