import base64, os
import datetime
import time
from datetime import date
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import pandas as pd
from docx import Document


from flask import Flask, send_from_directory
from flask import make_response 

from urllib.parse import quote as urlquote
from app import app

from ensemble.model_prediction import CNN_LSTM_Predictor
predictor = CNN_LSTM_Predictor()

####### download function 
@app.server.route("/download/<filename>", methods=['GET'])
def download(filename):
    directory = os.getcwd()  
    response = make_response(send_from_directory(directory, filename, as_attachment=True))
    #response = make_response(send_file(filename, as_attachment=True))
    response.headers["Content-Disposition"] = "attachment; filename={}".format(filename.encode().decode('latin-1'))
    return response


layout = html.Div([
    html.H6('请上传检测文档',style={
            'backgroundColor': '#00A1DF',
            'fontFamily':'Arial',
            'fontWeight': 'bold',
            'textAlign': 'left',
            'color': 'white','size':'5px','margin':'-2px'
            }),
    ## Progress bar
    # dbc.Progress(id = 'progress', value = 0, striped = True, animated =True),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files'),
            ' (* .docx files only) '
        ]),
        style={
            'width': '97%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='up-files'),
    html.Button('Submit',n_clicks=0, id='run-button',style={'float':'right','margin-right':'50px'}),
    html.Br(),
    html.Div([
        
        html.Div(id='select-file'),
        html.Div(id='detailed-table'),

        html.Br(),
        html.H6('汇总',style={
                    'backgroundColor': '#00A1DF',
                    'fontFamily':'Arial',
                    'fontWeight': 'bold',
                    'textAlign': 'left',
                    'color': 'white','size':'5px','margin':'-2px'}),
        html.Br(),
        html.Div(id='row5',style={'top-margin':'5px','bottom-margin':'5px'}),
        html.Div(id='row0',style={'top-margin':'5px','bottom-margin':'5px'}),
        html.Div(id='row1',style={'top-margin':'5px','bottom-margin':'5px'}),
        html.Div(id='row2',style={'top-margin':'5px','bottom-margin':'5px'}),
        html.Div(id='row3',style={'top-margin':'5px','bottom-margin':'5px'}),
        html.Div(id='row4',style={'top-margin':'5px','bottom-margin':'5px'}),
        html.H6('''
                    Please click the delete button after you downloaded all files you need to save space.
                ''', style = {'display': 'inline-block', 'color': '#0099ff'}),
        html.H6('''S''', style = {'display': 'inline-block', 'opacity': '0.0'}),
        html.Button('Delete', id='del_bttn', 
                            style = {'display': 'inline-block', 'color': '#0099ff',  'border-color': '#9fbfdf', 
                                     'font-size': '15px', 'font-family': 'Arial', 
                                     'width': '80px', 'height': '40px', 'padding': '1px'}),
        html.H6('''S''', style = {'display': 'inline-block', 'opacity': '0.0'}),
        html.Div(id='del_info', style = {'display': 'inline-block', 'color': '#0099ff'}),
        ],id='rest',style={'display':'none'}),

])

# Output what has been uploaded
@app.callback(Output('up-files', 'children'),
              [Input('upload-data', 'filename')]
             )   
def up_files(flnm):
    if flnm is not None:
        return 'File {} have been uploaded'.format(flnm)


## submit button for showing the rest
@app.callback(
    Output(component_id='rest', component_property='style'),
    [Input(component_id='run-button', component_property='n_clicks')]
)
def update_output(n_clicks):
    if n_clicks>0:
        return {'display': 'block'}
    else:
        raise PreventUpdate


def parse_contents(contents, filename):

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'docx' in filename:
            con = io.BytesIO(decoded)
            df = predictor.output_df(con)
            return df,filename,html.P(date.fromtimestamp(time.time()))

        else:
            df = pd.DataFrame()
            return df,filename,html.P(date.fromtimestamp(time.time()))

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])



############# interaction callback

############# filename radio items
@app.callback(Output('select-file', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               ])
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        file_type=all(['docx'in i for i in list_of_names])
        if file_type == True:
            options = []
            for i in range(0,len(list_of_names)):
                #print(list_of_names[i])
                options.append({'label':list_of_names[i],'value':str(i)},)

            children = dcc.RadioItems(
                options = options,
                value = '0',
                labelStyle={'display':'inline-block','color': '#005487','fontWeight': 'bold','font-size': '1.2em','text-align': 'justify','margin-right':'100px'},
                style = {'width':'150%'},
                id='file-options',
                )
            return  children 
        else:
            children = "Please upload .docx files only! Thanks!"
            return children


############### detailed table
df_columns = []
dt_list = []
@app.callback(Output('row5','children'),
             [Input('upload-data', 'contents')],
             [State('upload-data', 'filename')])
def outputDF(list_of_contents, list_of_names):
    # RESET global variables
    df_len = len(df_columns)
    if list_of_names:
        for i, n in enumerate(list_of_names):
            df, _, dt = parse_contents(list_of_contents[i], n)
            if i<df_len:
                df_columns[i]=df
                dt_list[i]=dt
            else:
                df_columns.append(df)
                dt_list.append(dt)
        return [f'{len(list_of_names)} file(s) in reviewing...']
    

@app.callback(Output('detailed-table','children'),
            [Input('file-options', 'value'),Input('row5','children')],
            [State('upload-data', 'contents'),State('upload-data', 'filename')])
def show_table(values, children, list_of_contents,list_of_names):
    ## TODO: save df_columns into tempfile in order to avoid running parse_content function 
    ## when value changes
    if children:
        fn_list = list_of_names
        val_int = int(values)
        index =  df_columns[val_int][df_columns[val_int]['labels']==1].index
        return html.Div([
            html.Br(),
            html.Div([html.P(fn_list[val_int]),
                    html.Div(dt_list[val_int],style={'display': 'inline-block','margin-left':'10px'})
                ],style={'display':'flex'}),

            dash_table.DataTable(
                id='datatable-birst-'+ values,
                data=df_columns[val_int].to_dict('records'),
                columns=[{"name": i, "id": i, 'deletable': False} for i in df_columns[val_int].columns[:-2]],
                editable=True,
                # filter_action="native",
                #sort_action="native",
                #sort_mode="multi",
                row_selectable="multi",
                #row_deletable=True,
                selected_rows=index,
                #page_action="native",
                #page_current= 0,
                #page_size= 10,
                style_table={'maxWidth': '1300px','maxHeight': '800px','overflowY': 'scroll','border': 'thin lightgrey solid'},
                style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'left','minWidth': '0px', 'maxWidth': '180px',
            'whiteSpace': 'normal'},
                css=[{
                    'selector': '.dash-cell div.dash-cell-value',
                    'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                }],
                
                ),    
            html.Button('Submit', id='editing-rows-button'+ values, n_clicks=0),
            ],style={'margin-left':'50px'})



############# detailed table feature----check the box, and turn yellow
@app.callback(Output('datatable-birst-0', 'style_data_conditional'),
     [Input('datatable-birst-0', 'selected_rows')])
def update_graphs(selected_rows):
    if selected_rows is not None:
        return [{
                 "if": {"row_index": 'odd'},
                'backgroundColor': 'rgb(248,248,248)',
                }] + [{
                 "if": {"row_index": c,
                'column_id': 'contents' },
                'backgroundColor': 'yellow',
                'color': 'black'}
                for c in selected_rows]

@app.callback(Output('datatable-birst-1', 'style_data_conditional'),
     [Input('datatable-birst-1', 'selected_rows')])
def update_graphs(selected_rows):
    if selected_rows is not None:
        return [{
                 "if": {"row_index": 'odd'},
                'backgroundColor': 'rgb(248,248,248)',
                }] + [{
                 "if": {"row_index": c,
                'column_id': 'contents' },
                'backgroundColor': 'yellow',
                'color': 'black'}
                for c in selected_rows]

@app.callback(Output('datatable-birst-2', 'style_data_conditional'),
     [Input('datatable-birst-2', 'selected_rows')])
def update_graphs(selected_rows):
    if selected_rows is not None:
        return [{
                 "if": {"row_index": 'odd'},
                'backgroundColor': 'rgb(248,248,248)',
                }] + [{
                 "if": {"row_index": c,
                'column_id': 'contents' },
                'backgroundColor': 'yellow',
                'color': 'black'}
                for c in selected_rows]

@app.callback(Output('datatable-birst-3', 'style_data_conditional'),
     [Input('datatable-birst-3', 'selected_rows')])
def update_graphs(selected_rows):
    if selected_rows is not None:
        return [{
                 "if": {"row_index": 'odd'},
                'backgroundColor': 'rgb(248,248,248)',
                }] + [{
                 "if": {"row_index": c,
                'column_id': 'contents' },
                'backgroundColor': 'yellow',
                'color': 'black'}
                for c in selected_rows]

@app.callback(Output('datatable-birst-4', 'style_data_conditional'),
     [Input('datatable-birst-4', 'selected_rows')])
def update_graphs(selected_rows):
      if selected_rows is not None:
        return [{
                 "if": {"row_index": 'odd'},
                'backgroundColor': 'rgb(248,248,248)',
                }] + [{
                 "if": {"row_index": c,
                'column_id': 'contents' },
                'backgroundColor': 'yellow',
                'color': 'black'}
                for c in selected_rows]      



#################### generate summary table function 
               
def generate_table(df, filename):
    col0 = filename
    col1 = date.fromtimestamp(time.time())
    c2 = sum(df['labels'])
    #c2 = sum([dic['labels'] for dic in df])
    col2 = 'Number of AEs (original) : ' + str(c2)
    col3 = 'Number of AEs (updated) : ' + str(c2)
    return pd.DataFrame({'File Names': [col0] ,\
                          'Date':[col1],\
                          '# of Model Predicted AEs': [col2],\
                          '# of Corrected AEs': [col3]}) 
 

 
############### summary row 0 + download 0
@app.callback(Output('row0', 'children'),
              [Input('row5','children')],
              [State('upload-data','filename')])
              
def updated_data(children,filenames):
    if children:
        if filenames is not None:
            dff = generate_table(df_columns[0], filenames[0])
            if len(dff) > 0 :
                row0_df = dff.copy()
                original_row0=[]
                for col in row0_df.columns:
                    value=row0_df[col]
                    original_row0.append(html.Td(value))

                return html.Div([
                    html.Tr(original_row0, style={
                            'fontFamily':'Arial',
                            'fontWeight': 'bold',
                            'textAlign': 'center',
                            'color': 'black',
                            'font_size':'16px',
                            }),
                    ], id='original_row0', style={'margin-left':'50px','display':'flex'})

    
######## updated row0
@app.callback(Output('original_row0', 'children'),
              [Input('datatable-birst-0', 'selected_rows'),Input('editing-rows-button0', 'n_clicks')],
              [State('upload-data', 'filename'),State('row5','children')])
def updated_data(selected_rows, n_clicks, filenames,children):
    if children:
        dff = generate_table(df_columns[0], filenames[0])
        row0_df = dff.copy()
        fn0 = filenames[0].split('.')[0]+'_AE.docx'
        if n_clicks > 0:
            new_row0=[]
            row0_df['# of Corrected AEs'] = 'Number of AEs (updated) : ' + str(len(selected_rows))
            for col in row0_df.columns:
                value=row0_df[col]
                new_row0.append(html.Td(value))

            return html.Div([
                html.Tr(new_row0,style={
                        'fontFamily':'Arial',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'color': 'black',
                        'font_size':'16px',
                        'background': 'rgb(248,248,248)',
                        }),
                html.Div([html.A("download", 
                    id='download0',
                    href="",
                    download= fn0,
                    n_clicks=0,
                    target="_blank",
                    )], style={'margin-left':'10px','margin-top':'10px','display':'inline-block'})
                ], style={'display':'flex'})
        else:
            original_row0=[]
            for col in row0_df.columns:
                value=row0_df[col]
                original_row0.append(html.Td(value))
            
            return html.Div([
                html.Tr(original_row0,style={
                        'fontFamily':'Arial',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'color': 'black',
                        'font_size':'16px',
                        }),
                ], style={'display':'flex'})
            

####### download row0


@app.callback(Output('download0', 'href'),
              [Input('datatable-birst-0', 'selected_rows')],
              [State('datatable-birst-0', 'data'),
               State('upload-data', 'contents'),
               State('upload-data', 'filename')])
def download_df(selected_rows,df, list_of_contents,filenames):
    ## create dataframe  -- data

    data = pd.DataFrame.from_dict(df, 'columns')
    updated_label=[]
    for i in range(len(data)):
        if i in selected_rows:
            updated_label.append(1)
        else:
            updated_label.append(0)
    data['labels'] = updated_label

    ## process uploaded document --doc
    contents = list_of_contents[0]
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    con = io.BytesIO(decoded)

    doc = predictor.tableMap2Docx(con, data)
    flnm = filenames[0].split('.')[0] + '_AE.docx'
    doc.save(flnm)

    location = "/download/{}".format(urlquote(flnm))

    return location



################### summary row1 + download 1
@app.callback(Output('row1', 'children'),
              [Input('row5','children')],
              [State('upload-data','filename')])
def updated_data(children,filenames):
    if children:
        if len(filenames)>1:
            dff = generate_table(df_columns[1], filenames[1])
            if len(dff) > 0 :
                row1_df = dff.copy()
                original_row1=[]
                for col in row1_df.columns:
                    value=row1_df[col]
                    original_row1.append(html.Td(value))

                return html.Div([
                    html.Tr(original_row1, style={
                            'fontFamily':'Arial',
                            'fontWeight': 'bold',
                            'textAlign': 'center',
                            'color': 'black',
                            'font_size':'16px',
                            }),
                    ], id='original_row1', style={'margin-left':'50px','display':'flex'})

    
######## updated row1
@app.callback(Output('original_row1', 'children'),
              [Input('datatable-birst-1', 'selected_rows'),Input('editing-rows-button1', 'n_clicks')],
              [State('upload-data', 'filename'),State('row5','children')])
def updated_data(selected_rows, n_clicks, filenames,children):
    if children:
        dff = generate_table(df_columns[1], filenames[1])
        row1_df = dff.copy()
        fn0 = filenames[1].split('.')[0]+'_AE.docx'
        if n_clicks > 0:
            new_row1=[]
            row1_df['# of Corrected AEs'] = 'Number of AEs (updated) : ' + str(len(selected_rows))
            for col in row1_df.columns:
                value=row1_df[col]
                new_row1.append(html.Td(value))

            return html.Div([
                html.Tr(new_row1,style={
                        'fontFamily':'Arial',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'color': 'black',
                        'font_size':'16px',
                        'background': 'rgb(248,248,248)',
                        }),
                html.Div([html.A("download", 
                    id='download1',
                    href="",
                    download= fn0,
                    n_clicks=0,
                    target="_blank",
                    )], style={'margin-left':'10px','margin-top':'10px','display':'inline-block'})
                ], style={'display':'flex'})
        else:
            original_row1=[]
            for col in row1_df.columns:
                value=row1_df[col]
                original_row1.append(html.Td(value))
            
            return html.Div([
                html.Tr(original_row1,style={
                        'fontFamily':'Arial',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'color': 'black',
                        'font_size':'16px',
                        }),
                ], style={'display':'flex'})
        

####### download row1
@app.callback(Output('download1', 'href'),
              [Input('datatable-birst-1', 'selected_rows')],
              [State('datatable-birst-1', 'data'),
               State('upload-data', 'contents'),
               State('upload-data', 'filename')])
def download_df(selected_rows,df, list_of_contents,filenames):
    ## create dataframe  -- data

    data = pd.DataFrame.from_dict(df, 'columns')
    updated_label=[]
    for i in range(len(data)):
        if i in selected_rows:
            updated_label.append(1)
        else:
            updated_label.append(0)
    data['labels'] = updated_label

    ## process uploaded document --doc
    contents = list_of_contents[1]
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    con = io.BytesIO(decoded)

    doc = predictor.tableMap2Docx(con, data)
    flnm = filenames[1].split('.')[0] + '_AE.docx'
    doc.save(flnm)

    location = "/download/{}".format(urlquote(flnm))

    return location



############### summary row 2 + download 2
@app.callback(Output('row2', 'children'),
              [Input('row5','children')],
              [State('upload-data','filename')])
def updated_data(children,filenames):
    if children:
        if len(filenames)>2:
            dff = generate_table(df_columns[2], filenames[2])
            if len(dff) > 0 :
                row2_df = dff.copy()
                original_row2=[]
                for col in row2_df.columns:
                    value=row2_df[col]
                    original_row2.append(html.Td(value))

                return html.Div([
                    html.Tr(original_row2, style={
                            'fontFamily':'Arial',
                            'fontWeight': 'bold',
                            'textAlign': 'center',
                            'color': 'black',
                            'font_size':'16px',
                            }),
                    ], id='original_row2', style={'margin-left':'50px','display':'flex'})

    
######## updated row2
@app.callback(Output('original_row2', 'children'),
              [Input('datatable-birst-2', 'selected_rows'),Input('editing-rows-button2', 'n_clicks')],
              [State('upload-data', 'filename'),State('row5','children')])

def updated_data(selected_rows, n_clicks, filenames,children):
    if children:
        dff = generate_table(df_columns[2], filenames[2])

        row2_df = dff.copy()
        fn0 = filenames[2].split('.')[0]+'_AE.docx'
        if n_clicks > 0:
            new_row2=[]
            row2_df['# of Corrected AEs'] = 'Number of AEs (updated) : ' + str(len(selected_rows))
            for col in row2_df.columns:
                value=row2_df[col]
                new_row2.append(html.Td(value))

            return html.Div([
                html.Tr(new_row2,style={
                        'fontFamily':'Arial',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'color': 'black',
                        'font_size':'16px',
                        'background': 'rgb(248,248,248)',
                        }),
                html.Div([html.A("download", 
                    id='download2',
                    href="",
                    download= fn0,
                    n_clicks=0,
                    target="_blank",
                    )], style={'margin-left':'10px','margin-top':'10px','display':'inline-block'})
                ], style={'display':'flex'})
        else:
            original_row2=[]
            for col in row2_df.columns:
                value=row2_df[col]
                original_row2.append(html.Td(value))
            
            return html.Div([
                html.Tr(original_row2,style={
                        'fontFamily':'Arial',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'color': 'black',
                        'font_size':'16px',
                        }),
                ], style={'display':'flex'})
        

####### download row2
@app.callback(Output('download2', 'href'),
              [Input('datatable-birst-2', 'selected_rows')],
              [State('datatable-birst-2', 'data'),
               State('upload-data', 'contents'),
               State('upload-data', 'filename')])
def download_df(selected_rows,df, list_of_contents,filenames):
    ## create dataframe  -- data

    data = pd.DataFrame.from_dict(df, 'columns')
    updated_label=[]
    for i in range(len(data)):
        if i in selected_rows:
            updated_label.append(1)
        else:
            updated_label.append(0)
    data['labels'] = updated_label

    ## process uploaded document --doc
    contents = list_of_contents[2]
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    con = io.BytesIO(decoded)

    doc = predictor.tableMap2Docx(con, data)
    flnm = filenames[2].split('.')[0] + '_AE.docx'
    doc.save(flnm)

    location = "/download/{}".format(urlquote(flnm))

    return location




############### summary row 3 + download 3
@app.callback(Output('row3', 'children'),
              [Input('row5','children')],
              [State('upload-data','filename')])
def updated_data(children,filenames):
    if children:
        if len(filenames)>3:
            dff = generate_table(df_columns[3], filenames[3])
            if len(dff) > 0 :
                row3_df = dff.copy()
                original_row3=[]
                for col in row3_df.columns:
                    value=row3_df[col]
                    original_row3.append(html.Td(value))

                return html.Div([
                    html.Tr(original_row3, style={
                            'fontFamily':'Arial',
                            'fontWeight': 'bold',
                            'textAlign': 'center',
                            'color': 'black',
                            'font_size':'16px',
                            }),
                    ], id='original_row3', style={'margin-left':'50px','display':'flex'})

    
######## updated row3
@app.callback(Output('original_row3', 'children'),
              [Input('datatable-birst-3', 'selected_rows'),Input('editing-rows-button3', 'n_clicks')],
              [State('upload-data', 'filename'),State('row5','children')])
def updated_data(selected_rows, n_clicks, filenames,children):
    if children:
        dff = generate_table(df_columns[3], filenames[3])
        row3_df = dff.copy()
        fn0 = filenames[3].split('.')[0]+'_AE.docx'
        if n_clicks > 0:
            new_row3=[]
            row3_df['# of Corrected AEs'] = 'Number of AEs (updated) : ' + str(len(selected_rows))
            for col in row3_df.columns:
                value=row3_df[col]
                new_row3.append(html.Td(value))

            return html.Div([
                html.Tr(new_row3,style={
                        'fontFamily':'Arial',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'color': 'black',
                        'font_size':'16px',
                        'background': 'rgb(248,248,248)',
                        }),
                html.Div([html.A("download", 
                    id='download3',
                    href="",
                    download= fn0,
                    n_clicks=0,
                    target="_blank",
                    )], style={'margin-left':'10px','margin-top':'10px','display':'inline-block'})
                ], style={'display':'flex'})
        else:
            original_row3=[]
            for col in row3_df.columns:
                value=row3_df[col]
                original_row3.append(html.Td(value))
            
            return html.Div([
                html.Tr(original_row3,style={
                        'fontFamily':'Arial',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'color': 'black',
                        'font_size':'16px',
                        }),
                ], style={'display':'flex'})
            

####### download row3
@app.callback(Output('download3', 'href'),
              [Input('datatable-birst-3', 'selected_rows')],
              [State('datatable-birst-3', 'data'),
               State('upload-data', 'contents'),
               State('upload-data', 'filename')])
def download_df(selected_rows,df, list_of_contents,filenames):
    ## create dataframe  -- data

    data = pd.DataFrame.from_dict(df, 'columns')
    updated_label=[]
    for i in range(len(data)):
        if i in selected_rows:
            updated_label.append(1)
        else:
            updated_label.append(0)
    data['labels'] = updated_label

    ## process uploaded document --doc
    contents = list_of_contents[3]
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    con = io.BytesIO(decoded)

    doc = predictor.tableMap2Docx(con, data)
    flnm = filenames[3].split('.')[0] + '_AE.docx'
    doc.save(flnm)

    location = "/download/{}".format(urlquote(flnm))

    return location



############### summary row 4 + download 4
@app.callback(Output('row4', 'children'),
              [Input('row5','children')],
              [State('upload-data','filename')])
def updated_data(children,filenames):
    if children:
        if len(filenames)>4:
            dff = generate_table(df_columns[4], filenames[4])

            if len(dff) > 0 :
                row4_df = dff.copy()
                original_row4=[]
                for col in row4_df.columns:
                    value=row4_df[col]
                    original_row4.append(html.Td(value))

                return html.Div([
                    html.Tr(original_row4, style={
                            'fontFamily':'Arial',
                            'fontWeight': 'bold',
                            'textAlign': 'center',
                            'color': 'black',
                            'font_size':'16px',
                            }),
                    ], id='original_row4', style={'margin-left':'50px','display':'flex'})

    
######## updated row4
@app.callback(Output('original_row4', 'children'),
              [Input('datatable-birst-4', 'selected_rows'),Input('editing-rows-button4', 'n_clicks')],
              [State('upload-data', 'filename'),State('row5','children')])
def updated_data(selected_rows, n_clicks, filenames,children):
    if children:
        dff = generate_table(df_columns[4], filenames[4])
        row4_df = dff.copy()
        fn0 = filenames[4].split('.')[0]+'_AE.docx'
        if n_clicks > 0:
            new_row4=[]
            row4_df['# of Corrected AEs'] = 'Number of AEs (updated) : ' + str(len(selected_rows))
            for col in row4_df.columns:
                value=row4_df[col]
                new_row4.append(html.Td(value))

            return html.Div([
                html.Tr(new_row4,style={
                        'fontFamily':'Arial',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'color': 'black',
                        'font_size':'16px',
                        'background': 'rgb(248,248,248)',
                        }),
                html.Div([html.A("download", 
                    id='download4',
                    href="",
                    download= fn0,
                    n_clicks=0,
                    target="_blank",
                    )], style={'margin-left':'10px','margin-top':'10px','display':'inline-block'})
                ], style={'display':'flex'})
        else:
            original_row4=[]
            for col in row4_df.columns:
                value=row4_df[col]
                original_row4.append(html.Td(value))
            
            return html.Div([
                html.Tr(original_row4,style={
                        'fontFamily':'Arial',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'color': 'black',
                        'font_size':'16px',
                        }),
                ], style={'display':'flex'})
        

####### download row4
@app.callback(Output('download4', 'href'),
              [Input('datatable-birst-4', 'selected_rows')],
              [State('datatable-birst-4', 'data'),
               State('upload-data', 'contents'),
               State('upload-data', 'filename')])
def download_df(selected_rows,df, list_of_contents,filenames):
    ## create dataframe  -- data

    data = pd.DataFrame.from_dict(df, 'columns')
    updated_label=[]
    for i in range(len(data)):
        if i in selected_rows:
            updated_label.append(1)
        else:
            updated_label.append(0)
    data['labels'] = updated_label

    ## process uploaded document --doc
    contents = list_of_contents[4]
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    con = io.BytesIO(decoded)

    doc = predictor.tableMap2Docx(con, data)
    flnm = filenames[4].split('.')[0] + '_AE.docx'
    doc.save(flnm)

    location = "/download/{}".format(urlquote(flnm))

    return location


# Output delete information
@app.callback(Output('del_info', 'children'),
              [Input('del_bttn', 'n_clicks')],
              [State('upload-data', 'filename')])
def del_fl(n_click, flnms): 
    if flnms is not None:
        for i in range(len(flnms)):
            file = flnms[i].split('.')[0]+'_AE.docx'
            if os.path.exists(file):
                os.remove(file)
        return 'DONE'

# if __name__ == '__main__':
#     app.run_server(debug=True)


