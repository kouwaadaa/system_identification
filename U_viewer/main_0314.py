# -*- coding: utf-8 -*-
'''
author: ub
Uviewer main．
'''

#---------------------------
# モジュールのインポートなど
#---------------------------
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd

#---------------------------
# ファイルを読み込んでデータファイルに格納
#---------------------------

# CSVファイルの読み込み
df = pd.read_csv(
    filepath_or_buffer="../log_data/20190314-2.csv",
    encoding='ASCII',
    sep=',',
    header=0,
    usecols=['ATT_Roll',
             'ATT_Pitch',
             'ATT_Yaw',
             'ATT_RollRate',
             'ATT_PitchRate',
             'ATT_YawRate',
             'LPOS_X',
             'LPOS_Y',
             'LPOS_Z',
             'LPOS_VX',
             'LPOS_VY',
             'LPOS_VZ',
             'GPS_Alt',
             'OUT0_Out0',
             'OUT0_Out1',
             'OUT0_Out2',
             'OUT0_Out3',
             'OUT0_Out4',
             'OUT0_Out5',
             'OUT1_Out0',
             'OUT1_Out1',
             'AIRS_TrueSpeed',
             'MAN_pitch',
             'MAN_roll',
             'MAN_yaw',
             'MAN_thrust',
             'VTOL_Tilt',
             'TIME_StartTime'
             ]
)

# 空白行を削除
df = df.dropna(how='all')
df = df.reset_index(drop=True)

# 時間データを[秒]に変換
df['Time_ST'] = df.at[0,'TIME_StartTime']
df['Time_sec'] = (df['TIME_StartTime'] - df['Time_ST'])/1000000

#---------------------------
# ここからアプリの設定と出力
#---------------------------

# アプリ起動
app = dash.Dash()

# ページ全体
app.layout = html.Div(
    [
        # トップ
        html.H1(
            children='U viewer',
            style={
                'textAlign': 'center',
            }
        ),
        # ドロップダウンメニュー
        html.Div(
            [
                dcc.Dropdown(
                    id='yaxis-dropdown',
                    options=[
                        {'label': 'Attitude', 'value': 'ATT'},
                        {'label': 'Position & Speed', 'value': 'LPOS'},
                        {'label': 'Thrust Command', 'value': 'OUT0'},
                        {'label': 'Manual', 'value': 'MANUAL'},
                    ],
                    # 初期値
                    value='ATT'
                )
            ],
            # CSS
            style={
                'width': '40%',
                'display': 'inline-block',
                'margin-bottom': '80px',
            }
        ),
        # グラフプロット
        html.Div(
            [
                dcc.Graph(
                    id='plot-data',
                    style={'height': '100vh'},
                )
            ],
            style={
            }
        ),
        # スライダー
        html.Div(
            [
                dcc.RangeSlider(
                    id='time-sec-slider',
                    min=df['Time_sec'].min(),
                    max=df['Time_sec'].max(),
                    value=[df['Time_sec'].min(), df['Time_sec'].max()],
                    step=1,
                    # 5刻みで数字を表示, 最大値は要修正...
                    marks={i*5: '{}'.format(i*5) for i in range(30)},
                )
            ],
            style={
                'width': '90%',
                'display': 'inline-block',
                'margin': '40px',
            }
        )
    ],
)

# 入力と出力を設定
@app.callback(
    Output('plot-data', 'figure'),
    [Input('yaxis-dropdown', 'value'),
     Input('time-sec-slider', 'value')])
# 入力2種類を受付ける．time_valueはlistなので注意．
def update_figure(selected_item,time_value):

    # スライダーの値に応じて切り取り
    dff = df[time_value[0] <= df['Time_sec']]
    dff = dff[dff['Time_sec'] <= time_value[1]]

    # データサイズ取り出し
    data_size = len(dff)

    # 姿勢角の表示
    if selected_item == 'ATT':
        traces=[
            go.Scatter(
                x=dff['Time_sec'],
                y=dff['ATT_Roll'],
                mode='lines',
                name='ATT_Roll'
            ),
            go.Scatter(
                x=dff['Time_sec'],
                y=dff['ATT_Pitch'],
                mode='lines',
                name='ATT_Pitch'
            ),
            go.Scatter(
                x=dff['Time_sec'],
                y=dff['ATT_Yaw'],
                mode='lines',
                name='ATT_Yaw'
            ),
        ]
        # 実際の表示
        return {
            'data': traces,
            'layout': go.Layout(
                xaxis={'title': 'Time[sec]'},
                yaxis={'title': selected_item},
            ),
            go.layout()
        }
    elif selected_item == 'LPOS':
        traces=[
            go.Scatter3d(
                x=dff['LPOS_X'],
                y=dff['LPOS_Y'],
                z=-dff['LPOS_Z'],
                mode='lines',
                name='Trace of UAV'
            ),
        ]
        # 実際の表示
        return {
            'data': traces,
            'layout': go.Layout(
                updatemenus= [{
                    'type': 'buttons',
                    'buttons': [{
                        'label': 'Flight Start',
                        'method': 'animate',
                        'args': [None]
                    }]
                }]
            ),
            'frames': [dict(
                data=[dict(
                    x=[dff['LPOS_X'][k]],
                    y=[dff['LPOS_Y'][k]],
                    z=[dff['LPOS_Z'][k]],
                    mode='markers',
                    marker=dict(color='red', size=8)
                )]
            ) for k in range(data_size)]
        }

    # スラスト指令値の表示
    elif selected_item == 'OUT0':
        traces=[
            go.Scatter(
                x=dff['Time_sec'],
                y=(dff['OUT0_Out0']-1000)/900,
                mode='lines',
                name='Tm_up'
            ),
            go.Scatter(
                x=dff['Time_sec'],
                y=(dff['OUT0_Out1']-1000)/900,
                mode='lines',
                name='Tm_down'
            ),
            go.Scatter(
                x=dff['Time_sec'],
                y=(dff['OUT0_Out2']-1000)/900,
                mode='lines',
                name='Tr_r'
            ),
            go.Scatter(
                x=dff['Time_sec'],
                y=(dff['OUT0_Out3']-1000)/900,
                mode='lines',
                name='Tr_l'
            ),
            go.Scatter(
                x=dff['Time_sec'],
                y=(dff['OUT0_Out4']-1000)/900,
                mode='lines',
                name='Tf_up'
            ),
            go.Scatter(
                x=dff['Time_sec'],
                y=(dff['OUT0_Out5']-1000)/900,
                mode='lines',
                name='Tf_down'
            )

        ]
        # 実際の表示
        return {
            'data': traces,
            'layout': go.Layout(
                xaxis={'title': 'Time[sec]'},
                yaxis={'title': selected_item},
            )
        }
    # マニュアル操作指令値の表示
    elif selected_item == 'MANUAL':
        traces=[
            go.Scatter(
                x=dff['Time_sec'],
                y=dff['MAN_roll'],
                mode='lines',
                name='Roll'
            ),
            go.Scatter(
                x=dff['Time_sec'],
                y=dff['MAN_pitch'],
                mode='lines',
                name='Pitch'
            ),
            go.Scatter(
                x=dff['Time_sec'],
                y=dff['MAN_yaw'],
                mode='lines',
                name='Yaw'
            ),
            go.Scatter(
                x=dff['Time_sec'],
                y=dff['MAN_thrust'],
                mode='lines',
                name='Thrust'
            ),
        ]
        # 実際の表示
        return {
            'data': traces,
            'layout': go.Layout(
                xaxis={'title': 'Time[sec]'},
                yaxis={'title': selected_item},
            )
        }
    # エラー回避
    else:
        return 1

# メイン実行
if __name__ == '__main__':
    app.run_server(debug=True)
