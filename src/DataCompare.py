from .Data import Data
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy import stats


class DataCompare(Data):
    def __init__(self,sql_path,project,dataset,table,
                date_focus, period,
                #  date_start,date_end,
                 dim_sql_date, dim_sql_index, dim_sql_dimensions,kpis,aggs,where_clause):

        super().__init__(sql_path, project, dataset, table,
                         date_focus, period, 
                         #  date_start, date_end, 
                         dim_sql_date, dim_sql_index, dim_sql_dimensions, kpis, aggs, where_clause)
        
        self._sql_path = "../src/sql/data_compare_pre_post.sql"
        self._project = project
        self._dataset = dataset
        self._table = table
        self._sql_file_string = super().get_sql_file_as_string()
        self._sql_variables = super().return_sql_variables()

        self._date_focus = date_focus
        self._period = period
        self._date_start, self._date_end = self.get_focus_date_range()
        self._d_start_comparison, self._d_end_comparison = self.get_comparison_date_range()

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

        self._pre_period = [self._d_start_comparison,self._d_end_comparison]
        self._post_period = [self._date_start,self._date_end]

        self._days_in_pre = datetime.strptime(self._d_end_comparison, '%Y-%m-%d').date() - datetime.strptime(self._d_start_comparison, '%Y-%m-%d').date()
        self._days_in_post = datetime.strptime(self._date_end, '%Y-%m-%d').date() - datetime.strptime(self._date_start, '%Y-%m-%d').date() 

        self._sql_file_string_formatted = super().get_sql_formatted()



    def get_focus_date_range(self):
        """
        Calculate the start and end dates of a specified time period (week, month, quarter, or year)
        based on the provided date and period.

        Args:
            self._date_focus (str): A date string in the format 'YYYY-MM-DD'.
            self._period (str): A string specifying the desired time period ('week', 'month', 'quarter', or 'year').

        Returns:
            tuple: A tuple containing two date strings in the format 'YYYY-MM-DD', representing the start and end dates
            of the specified time period.

        Raises:
            ValueError: If an invalid period is provided.

        Example:
            date_string = "2023-09-04"
            period = "month"
            start_date, end_date = get_date_range(date_string, period)
            # Returns ('2023-09-01', '2023-09-30')
        """
        # Convert the input date string to a datetime object
        date_object = datetime.strptime(self._date_focus, "%Y-%m-%d")

        if self._period == "week":
            # Find the weekday of the input date (0 = Monday, 6 = Sunday)
            weekday = date_object.weekday()
            # Calculate the Sunday week start date by subtracting the weekday
            start_date = date_object - timedelta(days=weekday)
            # Calculate the end date of the week (Saturday)
            end_date = start_date + timedelta(days=6)
        elif self._period == "month":
            # Calculate the first day of the current month
            start_date = date_object.replace(day=1)
            # Calculate the last day of the current month
            next_month = start_date.replace(month=start_date.month + 1)
            end_date = next_month - timedelta(days=1)
        elif self._period == "quarter":
            # Calculate the first day of the current quarter
            quarter_start_month = (date_object.month - 1) // 3 * 3 + 1
            start_date = date_object.replace(month=quarter_start_month, day=1)
            # Calculate the last day of the current quarter
            next_quarter = start_date.replace(month=start_date.month + 3)
            end_date = next_quarter - timedelta(days=1)
        elif self._period == "year":
            # Calculate the first day of the current year
            start_date = date_object.replace(month=1, day=1)
            # Calculate the last day of the current year
            next_year = start_date.replace(year=start_date.year + 1)
            end_date = next_year - timedelta(days=1)
        else:
            raise ValueError("Invalid period. Use 'week', 'month', 'quarter', or 'year'.")

        # Format the results as YYYY-MM-DD
        start_date_string = start_date.strftime("%Y-%m-%d")
        end_date_string = end_date.strftime("%Y-%m-%d")
        
        return start_date_string, end_date_string
    
    def get_comparison_date_range(self):
        """
        Calculate the start and end dates of the previous time period (week, month, quarter, or year)
        based on the provided date range (_date_start and _date_end).

        Returns:
            tuple: A tuple containing two date strings in the format 'YYYY-MM-DD', representing the start and end dates
            of the previous time period.

        Raises:
            ValueError: If an invalid period is provided.

        Example:
            _date_start = "2023-09-01"
            _date_end = "2023-09-30"
            period = "month"
            start_date, end_date = get_previous_date_range(_date_start, _date_end, period)
            # Returns ('2023-08-01', '2023-08-31')
        """
        
        # Convert the date strings to datetime objects
        date_start = datetime.strptime(self._date_start, "%Y-%m-%d")
        date_end = datetime.strptime(self._date_end, "%Y-%m-%d")

        if self._period == "week":
            # Calculate the start date of the previous week
            start_date = date_start - timedelta(weeks=1)
            # Calculate the end date of the previous week
            end_date = date_end - timedelta(weeks=1)
        elif self._period == "month":
            # Calculate the start date of the previous month
            start_date = date_start - relativedelta(months=1)

            # Calculate the end date of the previous month
            end_date = date_end.replace(day=1) - relativedelta(days=1)
        elif self._period == "quarter":
            # Calculate the start date of the previous quarter
            start_date = date_start - relativedelta(months=3)

            # Calculate the end date of the previous month
            end_date = date_end - relativedelta(months=3)
        elif self._period == "year":
            # Calculate the start date of the previous year
            start_date = date_start.replace(year=date_start.year - 1, month=1, day=1)
            # Calculate the end date of the previous year
            end_date = date_end.replace(year=date_end.year - 1, month=12, day=31)
        else:
            raise ValueError("Invalid period. Use 'week', 'month', 'quarter', or 'year'.")

        # Format the results as YYYY-MM-DD
        start_date_string = start_date.strftime("%Y-%m-%d")
        end_date_string = end_date.strftime("%Y-%m-%d")
        
        return start_date_string, end_date_string


# ██████╗ ██████╗ ███████╗    ██████╗  ██████╗ ███████╗████████╗     ██████╗ ██████╗ ███╗   ███╗██████╗  █████╗ ██████╗ ███████╗
# ██╔══██╗██╔══██╗██╔════╝    ██╔══██╗██╔═══██╗██╔════╝╚══██╔══╝    ██╔════╝██╔═══██╗████╗ ████║██╔══██╗██╔══██╗██╔══██╗██╔════╝
# ██████╔╝██████╔╝█████╗█████╗██████╔╝██║   ██║███████╗   ██║       ██║     ██║   ██║██╔████╔██║██████╔╝███████║██████╔╝█████╗  
# ██╔═══╝ ██╔══██╗██╔══╝╚════╝██╔═══╝ ██║   ██║╚════██║   ██║       ██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ██╔══██║██╔══██╗██╔══╝  
# ██║     ██║  ██║███████╗    ██║     ╚██████╔╝███████║   ██║       ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ██║  ██║██║  ██║███████╗
# ╚═╝     ╚═╝  ╚═╝╚══════╝    ╚═╝      ╚═════╝ ╚══════╝   ╚═╝        ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝

    def get_data_pre_post(self):
        """
        Extracts and separates data into pre-test and post-test periods based on date criteria.

        This method performs the following steps:
        1. Converts the 'date' column in the dataset to a datetime type for date comparison.
        2. Filters and extracts data points that occur on or before the specified 'self._date_test_prior'.
        This data represents the pre-test period.
        3. Filters and extracts data points that occur on or after the specified 'self._date_test'.
        This data represents the post-test period.
        4. Stores the pre-test data in 'self._data_pre' and the post-test data in 'self._data_post'.

        The separation of data into pre-test and post-test periods allows for subsequent analysis of the causal impact
        of an intervention or change on the specified dataset, typically in the context of A/B testing or similar studies.

        Note:
        - Ensure that the dataset (self._data_sql) and relevant date criteria (self._date_test_prior and self._date_test)
        are properly set before calling this method.

        Returns:
            None
        """
        # Convert 'date' column to datetime type for comparison
        self._data_sql[self._dim_sql_date] = pd.to_datetime(self._data_sql[self._dim_sql_date])

        pre_df = self._data_sql[self._data_sql[self._dim_sql_date] <= self._d_end_comparison]
        pre_df['focus_date'] = self._d_start_comparison
        pre_df['period'] = self._period

        post_df = self._data_sql[self._data_sql[self._dim_sql_date] >= self._date_start]
        post_df['focus_date'] = self._date_start
        post_df['period'] = self._period

        self._data_pre = pre_df
        self._data_post = post_df

    def get_data_pre_post_comparison(self, dimension_column: str, metric_column : str, metric_column_agg : str = 'sum'):
        """
        Performs data comparison and analysis between pre-test and post-test periods based on specified dimensions and metrics.

        Parameters:
            - dimension_column (str): The name of the dimension column used for grouping and comparison.
            - metric_column (str): The name of the metric column for which changes are analyzed.
            - metric_column_agg (str, optional): The aggregation method for the metric data (e.g., 'sum', 'mean').
            Default is 'sum'.

        This method performs the following steps:
        1. Converts the 'date' column in the dataset to a datetime type for date comparison.
        2. Filters and extracts data points that occur on or before the specified 'self._date_test_prior'.
        This data represents the pre-test period.
        3. Filters and extracts data points that occur on or after the specified 'self._date_test'.
        This data represents the post-test period.
        4. Aggregates the metric data for both pre-test and post-test periods based on the specified dimension column
        and aggregation method.
        5. Calculates the differences ('delta') between post-test and pre-test metric values.
        6. Calculates the percent changes ('%delta') between post-test and pre-test metric values as percentages.
        7. Returns a DataFrame containing the comparison results, including dimensions, metrics, deltas, and percent changes.

        This method enables the analysis of how a specified metric changes between the pre-test and post-test periods,
        broken down by different dimensions. It provides insights into the impact of an intervention or change
        on the specified metric.

        Note:
        - Ensure that the dataset (self._data_sql) and relevant date criteria (self._date_test_prior and self._date_test)
        are properly set before calling this method.

        Returns:
            pandas.DataFrame: A DataFrame containing the comparison results.
        """
        # Convert 'date' column to datetime type for comparison
        self._data_sql[self._dim_sql_date] = pd.to_datetime(self._data_sql[self._dim_sql_date])

        pre_df = self._data_sql[self._data_sql[self._dim_sql_date] <= self._d_end_comparison]
        post_df = self._data_sql[self._data_sql[self._dim_sql_date] >= self._date_start]

        #  aggregate the data based on dimensions or metrics, as specified.
        pre_grouped_by_dimension = pre_df.groupby(dimension_column).agg({metric_column: metric_column_agg})
        post_grouped_by_dimension = post_df.groupby(dimension_column).agg({metric_column: metric_column_agg})

        # Calculate the percent changes between the pre- and post-metrics using the aggregated DataFrames
        # percent_changes = (post_grouped_by_dimension - pre_grouped_by_dimension) / pre_grouped_by_dimension * 100


        # Merge the pre-aggregated, post-aggregated, and percent change DataFrames
        result_df = pre_grouped_by_dimension.merge(post_grouped_by_dimension, left_index=True, right_index=True, suffixes=('_pre', '_post'), how="outer")
        # Calculate differences between pre and post columns
        result_df['delta'] = result_df[f'{metric_column}_post'] - result_df[f'{metric_column}_pre']

        # Calculate percent changes formatted as percentages
        result_df['%delta'] = (result_df[f"delta"]/result_df[f"{metric_column}_pre"])*100
        result_df['%delta'] = result_df['%delta'].apply(lambda x: f"{x:.2f}%")

        # result_df = result_df.merge(percent_changes, left_index=True, right_index=True)
        self._data_pre_post_comparison = result_df
        return result_df

    def get_improving_dimensions(self):
        return self._data_pre_post_comparison[self._data_pre_post_comparison[f'delta']>0].sort_values(by="delta",ascending=False)

    def get_declining_dimensions(self):
        return self._data_pre_post_comparison[self._data_pre_post_comparison[f'delta']<0].sort_values(by="delta",ascending=True)
    
    def get_static_dimensions(self):
        return self._data_pre_post_comparison[self._data_pre_post_comparison[f'delta']==0]


    def get_new_dimensions(self, metric):
        return self._data_pre_post_comparison[(self._data_pre_post_comparison[f'{metric}_pre'].isnull()==True) & (self._data_pre_post_comparison[f'{metric}_post'].notnull()==True)].sort_values(by=f"{metric}_post",ascending=False)

    def get_lost_dimensions(self, metric):
        return self._data_pre_post_comparison[(self._data_pre_post_comparison[f'{metric}_pre'].notnull()==True) & (self._data_pre_post_comparison[f'{metric}_post'].isnull()==True)].sort_values(by=f"{metric}_pre",ascending=False)


    def get_outliers_zscore(self):
        self._data_pre_post_comparison['delta'] = self._data_pre_post_comparison['delta'].fillna(0)  # Fill missing values with zeros
        self._data_pre_post_comparison['delta'] = self._data_pre_post_comparison['delta'].astype(float)

        # Calculate Z-scores for 'delta'
        self._data_pre_post_comparison['z_score'] = stats.zscore(self._data_pre_post_comparison['delta'])

        # Identify outliers based on Z-score threshold
        outliers = self._data_pre_post_comparison[self._data_pre_post_comparison['z_score'].abs() > 2]
        return outliers
    
    def get_outliers_iqr(self):
        # Calculate IQR for 'delta'
        Q1 = self._data_pre_post_comparison['delta'].quantile(0.25)
        Q3 = self._data_pre_post_comparison['delta'].quantile(0.75)
        IQR = Q3 - Q1

        # Identify outliers based on IQR
        outliers = self._data_pre_post_comparison[(self._data_pre_post_comparison['delta'] < (Q1 - 1.5 * IQR)) | (self._data_pre_post_comparison['delta'] > (Q3 + 1.5 * IQR))]
        return outliers



    
    def get_mix_shift():
        # Mix Shift (v2)
        # categorical pre-post dataframe required
        return