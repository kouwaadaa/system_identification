# -*- coding: utf-8 -*-
'''
author: ub
ログデータの閲覧用．
'''

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


# CSVファイルの読み込み
df = pd.read_csv(
    filepath_or_buffer="../log_data/Book8.csv",
    encoding='ASCII',
    sep=',',
    header=0,
    usecols=['ATT_Roll',
             'ATT_Pitch',
             # 'ATT_Yaw',
             # 'ATT_RollRate',
             'ATT_PitchRate',
             # 'ATT_YawRate',
             # 'LPOS_X',
             # 'LPOS_Y',
             # 'LPOS_Z',
             'LPOS_VX',
             'LPOS_VY',
             'LPOS_VZ',
             'GPS_Alt',
             # 'OUT0_Out0',
             # 'OUT0_Out1',
             # 'OUT0_Out2',
             # 'OUT0_Out3',
             # 'OUT0_Out4',
             # 'OUT0_Out5',
             # 'OUT1_Out0',
             # 'OUT1_Out1',
             # 'AIRS_TrueSpeed',
             # 'MAN_pitch',
             # 'MAN_thrust',
             # 'VTOL_Tilt',
             'TIME_StartTime'
             ]
)

# 空白行を削除
df = df.dropna(how='all')
df = df.reset_index(drop=True)

# 時間データを[秒]に変換
df['Time_ST'] = df.at[0,'TIME_StartTime']
df['Time_sec'] = (df['TIME_StartTime'] - df['Time_ST'])/1000000

df[['GPS_Alt','Time_sec']].plot.line(x='Time_sec')

# アプリの宣言
app = dash.Dash()

# HTMLの外観を定義
app.layout = html.Div(children=[
    html.H4(children='US Agriculture Exports (2011)'),
    generate_table(df)
])

if __name__ == "__main__":
    app.run_server(debug=True)
