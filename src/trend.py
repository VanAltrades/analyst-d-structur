import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
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


def get_anomalous_records_std(self,value,std=2):
    """
    Identifies anomalies in a time-series DataFrame based on a specified metric and standard deviation threshold.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing time-series data with a 'date' column and the specified metric column.
    value : str
        The name of the column representing the metric to be analyzed for anomalies.
    std : int or float, optional
        The number of standard deviations used to determine the anomaly threshold. Default is 2.

    Returns
    -------
    data_anomaly : pd.DataFrame
        A DataFrame with additional columns indicating anomalies for each data point:
        - 'average_day_value': Mean metric value for each day of the week.
        - 'average_day_value_std': Standard deviation of metric values for each day of the week.
        - 'average_day_value_std_min_bound': Lower bound for anomaly detection.
        - 'average_day_value_std_max_bound': Upper bound for anomaly detection.
        - 'anomaly': Boolean column indicating whether a data point is an anomaly (True) or not (False).

    Notes
    -----
    Anomalies are determined by comparing the metric value of each data point to the mean and standard deviation
    of metric values for the corresponding day of the week. Data points falling outside the specified number
    of standard deviations from the mean are considered anomalies.

    NaN values in the DataFrame are replaced with 0.

    Examples
    --------
    df = get_anomalous_records_std(df, "visits", std=2)
    # Identifies anomalies in the 'visits' metric column using a 2-standard-deviation threshold.
    """
    data = self._data_sql
    data[self._dim_sql_date] = pd.to_datetime(data[self._dim_sql_date])

    # Filter and sort the DataFrame
    data = data[[self._dim_sql_date, value]].sort_values(by=self._dim_sql_date, ascending=False)

    # Set timestamp as index
    data.set_index(self._dim_sql_date, inplace=True)

    # Add a column for the day name
    data['day_name'] = data.index.day_name()

    # Calculate mean and standard deviation for each day of the week
    grouped = data.groupby('day_name')
    data['average_day_value'] = grouped[value].transform('mean')
    data['average_day_value_std'] = grouped[value].transform('std')

    # Calculate the lower and upper bounds for anomalies
    data['average_day_value_std_min_bound'] = data['average_day_value'] - (std * data['average_day_value_std'])
    data['average_day_value_std_max_bound'] = data['average_day_value'] + (std * data['average_day_value_std'])

    # Define a function to set the 'anomaly' column
    def set_anomaly(row):
        return (row[value] > row['average_day_value_std_max_bound']) or (row[value] < row['average_day_value_std_min_bound'])

    # Add an 'anomaly' column based on the defined function
    data['anomaly'] = data.apply(set_anomaly, axis=1)

    # Replace NaN values with 0
    data_anomaly = data.fillna(0)

    return data_anomaly

@staticmethod
def get_anomaly_trend(df,metric):
    """
    Generates a time series plot with markers to highlight anomalies in a specified metric.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing time-series data, including a datetime index.
    metric : str
        The name of the column representing the metric to be plotted.

    Returns
    -------
    None
        The function displays the plot using Plotly Express.

    Notes
    -----
    This function creates a time series plot using Plotly Express, visualizing the specified metric's trend over time.
    Anomalies in the metric are marked with red markers on the plot for easy identification.

    Parameters:
    - df: The DataFrame containing the data.
    - metric: The name of the metric to be plotted.

    Example
    -------
    get_anomaly_trend(df, "visits")
    # Generates a time series plot for the 'visits' metric, highlighting anomalies in red.
    """
    fig = px.line(
            df,
            x=df.index,
            y=metric,
            title=f"{metric.title()} Anomalies",
            template = 'plotly_dark')
    # create list of outlier_dates
    outlier_dates = df[df['anomaly'] == True].index
    # obtain y value of anomalies to plot
    y_values = [df.loc[i][metric] for i in outlier_dates]
    fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers',
                    name = 'anomaly',
                    marker=dict(color='red',size=10)))
    fig.show()


# def get_timelapse():
#     return