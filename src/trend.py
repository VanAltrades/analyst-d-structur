import plotly.express as px
import numpy as np

# def resample_date():
#     # daily, monthly, yearly
#     return
def format_data_for_plotly(self):
    # if dimension column, replace None with a string " " for plotly to interpret
    self._data_sql[self._kpis] = self._data_sql[self._kpis].replace(np.nan,0)
    # if dimension column, replace None with a string " " for plotly to interpret 
    self._data_sql[self._dim_value_list] = self._data_sql[self._dim_value_list].replace(np.nan," ")
    self._data_sql = self._data_sql.sort_values(by=self._dim_sql_date, ascending=False)
    return self._data_sql

def get_trend_line(self, **kwargs):
    fig = px.line(data_frame=self._data_sql, **kwargs)
    return fig.show()

def get_trend_bar(self, **kwargs):
    fig = px.bar(data_frame=self._data_sql, **kwargs)
    return fig.show()

# def get_anomaly():
#     """
#     https://medium.com/@sztistvan/change-point-detection-in-time-series-using-chatgpt-22cc9172a130
#     https://pypi.org/project/ruptures/
#     """
#     return

# def show_anomaly():
#     return

# def get_timelapse():
#     return