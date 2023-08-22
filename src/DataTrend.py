from Data import Data
import plotly.express as px
import numpy as np

class DataTrend(Data):
    def __init__(self,sql_path,project,dataset,table,date_start,date_end,dim_sql_date, dim_sql_index, dim_sql_dimensions,kpis,aggs,where_clause):

        super().__init__(sql_path, project, dataset, table, date_start, date_end, dim_sql_date, dim_sql_index, dim_sql_dimensions, kpis, aggs, where_clause)
        
        self._sql_path = sql_path
        self._project = project
        self._dataset = dataset
        self._table = table
        self._sql_file_string = super().get_sql_file_as_string()
        self._sql_variables = super().return_sql_variables()
        self._date_start = date_start
        self._date_end = date_end
        self._dim_sql_date = dim_sql_date
        self._dim_sql_index = dim_sql_index
        self._dim_sql_dimensions = dim_sql_dimensions
        self._dim_dict = super().get_dim_item_dict()
        self._dim_key_list = super().get_dim_key_list()
        self._dim_value_list = super().get_dim_value_list()
        self._kpis = kpis
        self._aggs = aggs
        self._kpi_aggregates = super().get_kpi_aggregate_functions()
        self._where_clause = where_clause
        self._group_by_clause = super().get_dynamic_group_by_clause()
        self._sql_file_string_formatted = super().get_sql_formatted()
        # self._data_sql = super().get_data_sql()


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
        # sorted_data = self._data_sql.sort_values(by=self._dim_sql_date, ascending=False)
        fig = px.line(data_frame=self._data_sql, **kwargs)
        return fig.show()

    # def get_anomaly():
    #     """
    #     https://medium.com/@sztistvan/change-point-detection-in-time-series-using-chatgpt-22cc9172a130
    #     https://pypi.org/project/ruptures/
    #     """
    #     return

    # def show_anomaly():
    #     return

    # def show_trend():
    #     return

    # def show_timelapse():
    #     return