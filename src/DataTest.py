from .Data import Data
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import statsmodels.api as sm
# CausalImpact
from causalimpact import CausalImpact
# SyntheticControl
import SparseSC
import warnings
import plotly.graph_objects as pgo
pd.set_option("display.max_columns", None)
warnings.filterwarnings('ignore')
# https://towardsdatascience.com/causal-inference-using-difference-in-differences-causal-impact-and-synthetic-control-f8639c408268

class DataTest(Data):
    def __init__(self,sql_path,project,dataset,table,date_start,date_end,dim_sql_date, dim_sql_index, dim_sql_dimensions,kpis,aggs,where_clause,
    dim_test_index, regex, date_test, batch_table):

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

        # TEST instance variables
        self._dim_test_index=dim_test_index    # "url"
        self._regex=regex                      # ['product[0-9_.+-]+', '(/category/[0-9_.+-]+|/category/z[0-9_.+-]+)', 'midtail\/(.*?)\/', '\/brand\/(.*?)\/']
        self._date_test=date_test              # "2017-02-05"
        self._date_test_prior = self.get_date_test_prior()
        self._batch_tbl=batch_table          # "e-commerce-demo-v.tests.Batch1_product_content_2017-02-05"

        self._pre_period = [self._date_start,self._date_test_prior]
        self._post_period = [self._date_test,self._date_end]

        self._days_in_pre = datetime.strptime(self._date_test_prior, '%Y-%m-%d').date() - datetime.strptime(self._date_start, '%Y-%m-%d').date()
        self._days_in_post = datetime.strptime(self._date_end, '%Y-%m-%d').date() - datetime.strptime(self._date_test, '%Y-%m-%d').date() 

        self._sql_file_string_formatted = super().get_sql_formatted()

        # TODO: make unique test/control dataframe instance variables for comps

    def get_date_test_prior(self):
        # Convert input date string to a datetime object
        date_test = datetime.strptime(str(self._date_test), "%Y-%m-%d")
        # Subtract one day
        one_day = timedelta(days=1)
        previous_day = date_test - one_day
        # Format the result as "YYYY-MM-DD"
        return previous_day.strftime("%Y-%m-%d")        
    
    def estimate_time_to_significance(self, metric, effect_size, alpha=0.05, power=0.8):
        """
        Estimate the time required to reach significance for a treatment group based on historical data.

        Parameters:
        - df: DataFrame containing historic treatment metrics with 'date' and 'metric' columns.
        - effect_size: The desired effect size (expected change due to treatment).
        - alpha: Significance level (default is 0.05).
        - power: Desired statistical power (default is 0.8).

        Returns:
        - Estimated time (sample size) required to reach significance.

        Notes:
        - The DataFrame 'df' should have two columns: 'date' and 'metric', where 'date' represents the date of observation
        and 'metric' represents the metric of interest.

        - The function estimates the required sample size (time) for the desired power, alpha, and effect size.
        """
        df = self._data_sql
        
        # pre period test data
        date_test_prior = datetime.strptime(self._date_test_prior, "%Y-%m-%d")
        # Convert the 'self._dim_sql_date' column in the DataFrame to datetime format
        df[self._dim_sql_date] = pd.to_datetime(df[self._dim_sql_date])
        df = df.loc[(df["test_group"]=="Test")&(df[self._dim_sql_date]<=date_test_prior)]
        
        # Calculate standard deviation and sample size from historic data
        historic_std = df[metric].std()
        print(f"historic std. dev. of {metric}: {historic_std}")
        sample_size_historic = len(df)
        print(f"historic sample size {sample_size_historic}")

        # Calculate the required sample size (time) for the desired power and alpha
        required_sample_size = sm.stats.tt_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative='larger',  # Use 'larger' for one-sided test (increase)
            nobs=None,  # Number of observations (sample size) is the unknown
            # ratio=1,  # Assume 1:1 allocation to treatment and control
            # alternative='two-sided'  # Use 'two-sided' for a two-sided test
        )

        return required_sample_size

#  ██████╗ █████╗ ██╗   ██╗███████╗ █████╗ ██╗         ██╗███╗   ███╗██████╗  █████╗  ██████╗████████╗
# ██╔════╝██╔══██╗██║   ██║██╔════╝██╔══██╗██║         ██║████╗ ████║██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
# ██║     ███████║██║   ██║███████╗███████║██║         ██║██╔████╔██║██████╔╝███████║██║        ██║   
# ██║     ██╔══██║██║   ██║╚════██║██╔══██║██║         ██║██║╚██╔╝██║██╔═══╝ ██╔══██║██║        ██║   
# ╚██████╗██║  ██║╚██████╔╝███████║██║  ██║███████╗    ██║██║ ╚═╝ ██║██║     ██║  ██║╚██████╗   ██║   
#  ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝    ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═╝  

    def get_causal_impact(self, test_group="Test", metric_field="visits", agg="sum"):
        """
        Analyzes the causal impact of a specified metric for a given test group over time.

        Parameters:
            - test_group (str): The name of the test group to analyze. Default is "Test".
            - metric_field (str): The name of the metric field to analyze. Default is "visits".
            - agg (str): The aggregation method for the metric data (e.g., "sum", "mean"). Default is "sum".

        This method performs the following steps:
            1. Filters the dataset to select data points associated with the specified test group.
            2. Aggregates the metric data over time, using the specified aggregation method.
            3. Prepares the data for analysis, ensuring it has the correct data types and structure.
            4. Computes the causal impact analysis using the CausalImpact library.
            5. Prints a summary of the causal impact analysis, including statistical results.
            6. Plots the causal impact analysis to visualize the effects.
            7. Prints a report summarizing the causal impact analysis.

        The CausalImpact library is used to assess the causal effect of a change or intervention on the specified metric.
        The analysis is based on a pre-period and post-period comparison, and it provides insights into the impact of the
        intervention, along with statistical significance.

        Note:
        - Ensure that the dataset (self._data_sql) and relevant parameters are properly set before calling this method.
        - The results of the analysis are printed to the console for inspection.

        Returns:
            None
        """
        _df = self._data_sql.loc[self._data_sql['test_group']==test_group] 
        _df = _df.groupby(self._dim_sql_date, as_index=False).agg({metric_field:[agg]})
        _df = _df[[self._dim_sql_date,metric_field]]
        
        _df.set_index(self._dim_sql_date, inplace=True)
        _df.index = pd.DatetimeIndex(_df.index)
        _df[metric_field] = _df[metric_field].astype(float) # FLOAT64 solved ValueError: Pandas data cast to numpy dtype of object. Check input data with np.asarray(data)

        ci = CausalImpact(_df, self._pre_period, self._post_period)
        print(ci.summary())
        ci.plot()
        print(ci.summary(output='report'))

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

        pre_df = self._data_sql[self._data_sql[self._dim_sql_date] <= self._date_test_prior]
        post_df = self._data_sql[self._data_sql[self._dim_sql_date] >= self._date_test]

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

        pre_df = self._data_sql[self._data_sql[self._dim_sql_date] <= self._date_test_prior]
        post_df = self._data_sql[self._data_sql[self._dim_sql_date] >= self._date_test]

        #  aggregate the data based on dimensions or metrics, as specified.
        pre_grouped_by_dimension = pre_df.groupby(dimension_column).agg({metric_column: metric_column_agg})
        post_grouped_by_dimension = post_df.groupby(dimension_column).agg({metric_column: metric_column_agg})

        # Calculate the percent changes between the pre- and post-metrics using the aggregated DataFrames
        # percent_changes = (post_grouped_by_dimension - pre_grouped_by_dimension) / pre_grouped_by_dimension * 100


        # Merge the pre-aggregated, post-aggregated, and percent change DataFrames
        result_df = pre_grouped_by_dimension.merge(post_grouped_by_dimension, left_index=True, right_index=True, suffixes=('_pre', '_post'))
        # Calculate differences between pre and post columns
        result_df['delta'] = result_df[f'{metric_column}_post'] - result_df[f'{metric_column}_pre']

        # Calculate percent changes formatted as percentages
        result_df['%delta'] = (result_df[f"delta"]/result_df[f"{metric_column}_pre"])*100
        result_df['%delta'] = result_df['%delta'].apply(lambda x: f"{x:.2f}%")

        # result_df = result_df.merge(percent_changes, left_index=True, right_index=True)
        return result_df

    # @staticmethod
    # def get_new_dimensions(df_pre_post_comparison, metric):
    #     return df_pre_post_comparison[(df_pre_post_comparison[f'{metric}_post'].notnull()==True) & (df_pre_post_comparison[f'{metric}_pre'].isnull()==False)]

    # @staticmethod
    # def get_lost_dimensions(df_pre_post_comparison, metric):
    #     return df_pre_post_comparison[(df_pre_post_comparison[f'{metric}_pre'].notnull()==True) & (df_pre_post_comparison[f'{metric}_post'].isnull()==True)]

    # @staticmethod
    # def get_improving_dimensions(df_pre_post_comparison):
    #     return df_pre_post_comparison[df_pre_post_comparison[f'delta']>0]

    # @staticmethod
    # def get_declining_dimensions(df_pre_post_comparison):
    #     return df_pre_post_comparison[df_pre_post_comparison[f'delta']<=0]



    # def get_outliers_by_metrics():
    #     return

    # def get_outliers_by_dimensions():
    #     return

    # def get_test_group():
    #     return

    # def get_control_group():
    #     return


    
    # def get_diff_in_diff():
    #     return


    # ███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗███████╗████████╗██╗ ██████╗     ██████╗ ██████╗ ███╗   ██╗████████╗██████╗  ██████╗ ██╗     
    # ██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║██╔════╝╚══██╔══╝██║██╔════╝    ██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔══██╗██╔═══██╗██║     
    # ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║█████╗     ██║   ██║██║         ██║     ██║   ██║██╔██╗ ██║   ██║   ██████╔╝██║   ██║██║     
    # ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║██╔══╝     ██║   ██║██║         ██║     ██║   ██║██║╚██╗██║   ██║   ██╔══██╗██║   ██║██║     
    # ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║███████╗   ██║   ██║╚██████╗    ╚██████╗╚██████╔╝██║ ╚████║   ██║   ██║  ██║╚██████╔╝███████╗
    # ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝     ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
                                                                                                                                            

    def format_synthetic_control_df(self,col_date="date",col_metric="visits",col_breakout="breakout"):
        """
        Formats and prepares the dataset for synthetic control analysis.

        Parameters:
            - col_metric (str, optional): The name of the metric column to use in the analysis. Default is "visits".
            - col_breakout (str, optional): The name of the breakout (dimension) column to use. Default is "breakout".

        This method performs the following steps:
        1. Pivots the dataset to create a matrix where rows represent different breakouts (dimensions),
        columns represent dates, and the values are the specified metric.
        2. Fills any missing values (NaN) in the resulting matrix with zeros to ensure a complete dataset.
        3. Returns the formatted DataFrame suitable for use in synthetic control analysis.

        The purpose of this method is to transform the data into a format that is compatible with synthetic control
        analysis, which typically requires data organized in a matrix-like structure. This analysis technique is used
        to estimate the counterfactual effect of an intervention or treatment by comparing it to a weighted combination
        of similar units that did not receive the treatment.

        Note:
        - Ensure that the dataset (self._data_sql) and relevant column names are properly set before calling this method.

        Returns:
            pandas.DataFrame: A formatted DataFrame ready for synthetic control analysis.
        """
        # use sql_path="../src/sql/data_test_synthetic_control.sql"
        df = self._data_sql
        df[col_date] = pd.to_datetime(df[col_date], format='%Y-%m-%d')
        df = df.pivot(index=col_breakout, columns=self._dim_sql_date, values=col_metric)        
        df.fillna(0,inplace=True)
        return df

    def get_sythethic_control_object(self,test_id="Test",col_date="date",col_metric="visits",col_breakout="breakout"):
        """
        Constructs and returns a synthetic control object for causal analysis.

        Parameters:
            - test_id (str, optional): The identifier of the test group. Default is "Test".
            - col_metric (str, optional): The name of the metric column to use in the analysis. Default is "visits".
            - col_breakout (str, optional): The name of the breakout (dimension) column to use. Default is "breakout".

        This method performs the following steps:
        1. Calls the `format_synthetic_control_df()` method to format and prepare the dataset for synthetic control analysis.
        2. Constructs a synthetic control object using the SparseSC.fit_fast() method, which is a component
            of the synthetic control analysis library.
        3. The constructed synthetic control object is trained using the pre-intervention data of the specified
            test group, considering the specified metric and breakout dimensions.

        The purpose of this method is to create a synthetic control object that can be used to estimate the causal
        effect of an intervention or treatment (specified by the `test_id`) on the metric of interest. The synthetic
        control method compares the treated unit (test group) to a weighted combination of control units to infer
        the counterfactual outcome in the absence of the intervention.

        Note:
        - Ensure that the dataset (self._data_sql) and relevant parameters are properly set before calling this method.

        Returns:
            SparseSC: A synthetic control object ready for causal analysis.
        """

        df = self.format_synthetic_control_df(col_date,col_metric,col_breakout)
        # display(df.head())
        sc_new = SparseSC.fit_fast( 
            features=df.iloc[:,df.columns <= self._date_test].values,
            targets=df.iloc[:,df.columns > self._date_test].values,
            treated_units=[idx for idx, val in enumerate(df.index.values) if val == test_id]
        )
        return sc_new

    def plot_synthetic_control_gap(self, col_date="date",col_metric="visits",col_breakout="breakout", test_id="Test"):
        """
        Generates a plot to visualize the gap between the test group and synthetic control in terms of the specified metric.

        Parameters:
            
            - test_id (str, optional): The identifier of the test group. Default is "Test".

        This method performs the following steps:
        1. Selects data from the `synth_df` DataFrame corresponding to the test group (`test_id`) and the synthetic control.
        2. Calculates the mean of the synthetic control values for each date.
        3. Creates a line plot using Plotly Express (`px.line`) to visualize the trends of the test group and synthetic control.
        4. Adds a dashed vertical line to indicate the change event date (intervention date).
        5. Customizes the plot title, legend, axis labels, and font size.
        6. Displays the generated plot.

        The purpose of this method is to provide a visual representation of the gap between the test group and synthetic control
        in terms of a specified metric over time. This helps in understanding the impact of an intervention or treatment on the
        test group compared to the synthetic control.

        Note:
        - Ensure that the `synth_df` DataFrame and relevant parameters are properly set before calling this method.

        Returns:
            None
        """
        synth_df = self.format_synthetic_control_df(col_date,col_metric,col_breakout)

        plot_df = synth_df.loc[synth_df.index == test_id].T.reset_index(drop=False)
        plot_df["Synthetic_Control"] = synth_df.loc[synth_df.index != test_id].mean(axis=0).values


        fig = px.line(
                data_frame = plot_df, 
                x = self._dim_sql_date, 
                y = [test_id,"Synthetic_Control"], 
                template = "plotly_dark")

        fig.add_trace(
            pgo.Scatter(
                x=[self._date_test,self._date_test],
                y=[plot_df[test_id].min()*0.98,plot_df['Synthetic_Control'].max()*1.02], 
                line={
                    'dash': 'dash',
                }, name='Change Event'
            ))
        fig.update_layout(
                title  = {
                    'text':f"Gap in Test v. Synthetic Control {col_metric.title()}",
                    'y':0.95,
                    'x':0.5,
                },
                legend =  dict(y=1, x= 0.8, orientation='v'),
                legend_title = "legend",
                xaxis_title="Date", 
                yaxis_title=f"{col_metric.title()} Trend",
                font = dict(size=15)
        )
        fig.show(renderer='notebook')
        # return

    def get_synthetic_control_time_series_df(self, col_date="date",col_metric="visits",col_breakout="breakout", test_id="Test"):
        """
        Generates a time series DataFrame with observed and synthetic control values for the specified test group.

        Parameters:
            - synth_df (pandas.DataFrame): The DataFrame containing synthetic control and test group data.
            - test_id (str, optional): The identifier of the test group. Default is "Test".

        This method performs the following steps:
        1. Extracts features and targets from the `synth_df` DataFrame based on the specified test date.
        2. Constructs a synthetic control model using the SparseSC.fit_fast() method, considering the specified test group.
        3. Creates a time series DataFrame with columns for date, observed values, and synthetic control values.
        4. Populates the time series DataFrame with observed and synthetic control values.
        5. Converts observed and synthetic control values to integer data types.
        6. Returns the resulting time series DataFrame.

        The purpose of this method is to provide a time series representation of observed and synthetic control values for
        the specified test group. This allows for visualizing and comparing the actual and synthetic control trends over time.

        Note:
        - Ensure that the `synth_df` DataFrame and relevant parameters are properly set before calling this method.

        Returns:
            pandas.DataFrame: A time series DataFrame with observed and synthetic control values.
        """

        synth_df = self.format_synthetic_control_df(col_date,col_metric,col_breakout)
        
        date_array_as_datetime = pd.to_datetime(synth_df.columns)
        try:
            date_test = datetime.strptime(self._date_test, '%Y-%m-%d')
        except:
            date_test = datetime.datetime.strptime(self._date_test, '%Y-%m-%d')

        ## creating required features
        features = synth_df.iloc[:,date_array_as_datetime <= date_test].values
        targets = synth_df.iloc[:,date_array_as_datetime > date_test].values

        # Check if features and targets have the same number of rows
        if features.shape[0] != targets.shape[0]:
            raise ValueError("Features and targets must have the same number of rows.")
        
        treated_units = [idx for idx, val in enumerate(synth_df.index.values) if val == test_id] # [2]

        ## Fit fast model for fitting Synthetic controls
        try:
            # Fit the model
            sc_model = SparseSC.fit_fast(
                features=features,
                targets=targets,
                treated_units=treated_units
            )
        except Exception as e:
            raise ValueError("Error occurred during model fitting: {}".format(str(e)))

        synth_df_timeseries = synth_df.loc[synth_df.index == test_id].T.reset_index(drop=False)
        synth_df_timeseries.columns = ["date", "Observed"] 
        synth_df_timeseries['Synthetic_Control'] = sc_model.predict(synth_df.values)[treated_units,:][0]
        synth_df_timeseries['Observed'] = synth_df_timeseries['Observed'].astype(int)
        synth_df_timeseries['Synthetic_Control'] = synth_df_timeseries['Synthetic_Control'].astype(int)
        synth_df_timeseries['Test Effect'] = synth_df_timeseries['Observed'] - synth_df_timeseries['Synthetic_Control']
        return synth_df_timeseries        

    def plot_synthetic_control_assessment(self, synth_df_timeseries, col_date="date",col_metric="visits"):
        """
        Generates a plot to assess the performance of the synthetic control model.

        Parameters:
            - result (pandas.DataFrame): A time series DataFrame containing observed and synthetic control values.
            - col_metric (str, optional): The name of the metric column to visualize. Default is "visits".

        This function performs the following steps:
        1. Creates a line plot using Plotly Express (`px.line`) to visualize the observed and synthetic control values
        over time.
        2. Adds a dashed vertical line to indicate the change event date (intervention date).
        3. Customizes the plot title, legend, axis labels, and font size.
        4. Displays the generated plot for assessing the performance of the synthetic control model.

        The purpose of this function is to provide a visual assessment of how well the synthetic control model replicates
        the observed data. It allows for evaluating the effectiveness of the model in capturing the impact of an intervention
        or treatment on the specified metric.

        Note:
        - Ensure that the `result` DataFrame containing observed and synthetic control values is properly prepared
        before calling this function.

        Returns:
            None
        """
        fig = px.line(
                data_frame = synth_df_timeseries, 
                x = "date", 
                y = ["Observed","Synthetic_Control"], 
                template = "plotly_dark",)

        fig.add_trace(
            pgo.Scatter(
                x=[self._date_test,self._date_test],
                y=[synth_df_timeseries.Observed.min()*0.98,synth_df_timeseries.Observed.max()*1.02], 
                line={
                    'dash': 'dash',
                }, name='Change Event'
            ))
        fig.update_layout(
                title  = {
                    'text':"Synthetic Control Assessment",
                    'y':0.95,
                    'x':0.5,
                },
                legend =  dict(y=1, x= 0.8, orientation='v'),
                legend_title = "",
                xaxis_title=f"{col_date.title()}", 
                yaxis_title=f"{col_metric.title()}",
                font = dict(size=15)
        )
        fig.show(renderer='notebook')
        # return

    def plot_synthetic_control_difference_across_time(self, synth_df_timeseries,col_metric="impressions"):
        """
        Generates a plot to visualize the difference in metrics between observed and synthetic control values over time.

        Parameters:
            - result (pandas.DataFrame): A time series DataFrame containing observed and synthetic control values.

        This function performs the following steps:
        1. Computes the difference between the observed and synthetic control values at each time point and adds it as
        a new column named 'Test Effect' to the DataFrame.
        2. Creates a line plot using Plotly Express (`px.line`) to visualize the difference in metrics over time.
        3. Adds a horizontal line at the y-axis value of 0 to indicate the baseline (no difference) level.
        4. Adds a dashed vertical line to indicate the change event date (intervention date).
        5. Customizes the plot title, legend, axis labels, and font size.
        6. Displays the generated plot for assessing the difference in metrics across time.

        The purpose of this function is to provide a visual representation of how the observed metrics deviate from
        the synthetic control metrics over time. It allows for assessing the impact of an intervention or treatment
        on the specified metric.

        Note:
        - Ensure that the `result` DataFrame containing observed and synthetic control values is properly prepared
        before calling this function.

        Returns:
            None
        """
        #| code-fold: true
        #| fig-cap: Fig - Gap in Per-capita cigarette sales in California w.r.t Synthetic Control

        synth_df_timeseries['Test Effect'] = synth_df_timeseries['Observed'] - synth_df_timeseries['Synthetic_Control']
        fig = px.line(
                data_frame = synth_df_timeseries, 
                x = "date", 
                y = "Test Effect", 
                template = "plotly_dark",)
        fig.add_hline(0)
        fig.add_trace(
            pgo.Scatter(
                x=[self._date_test,self._date_test],
                y=[synth_df_timeseries["Test Effect"].min()*0.98,synth_df_timeseries["Test Effect"].max()*1.02], 
                line={
                    'dash': 'dash',
                }, name='Change Event'
            ))

        fig.update_layout(
                title  = {
                    'text':"Difference across time",
                    'y':0.95,
                    'x':0.5,
                },
                legend =  dict(y=1, x= 0.8, orientation='v'),
                legend_title = "legend",
                xaxis_title="Date", 
                yaxis_title=f"Gap in {col_metric.title()}",
                font = dict(size=15)
        )
        fig.show(renderer='notebook')
        # return

    def get_synthetic_control_treatment_effect(self, synth_df_timeseries, col_metric="visits"):
        """
        Calculates and prints the treatment effect of a change event with respect to the synthetic control.

        Parameters:
            - result (pandas.DataFrame): A time series DataFrame containing observed and synthetic control values.
            - col_metric (str, optional): The name of the metric column for which the treatment effect is calculated.
            Default is "visits".

        This function performs the following steps:
        1. Extracts the treatment effect value at the end of the time series, which represents the impact of a change event.
        2. Rounds the treatment effect value to one decimal place for clarity.
        3. Prints the treatment effect along with the specified metric column.

        The purpose of this function is to provide a summary of the treatment effect of a change event with respect to the
        synthetic control. It calculates and presents the impact of the intervention or treatment on the specified metric.

        Note:
        - Ensure that the `result` DataFrame containing observed and synthetic control values is properly prepared
        before calling this function.

        Returns:
            None
        """
        print(f"Effect of Change Event w.r.t Synthetic Control => {np.round(synth_df_timeseries.loc[synth_df_timeseries['date']==self._date_end,'Test Effect'].values[0],1)} {col_metric}")
        # return




    def get_synthetic_control():
        """
        https://towardsdatascience.com/causal-inference-using-difference-in-differences-causal-impact-and-synthetic-control-f8639c408268
        https://towardsdatascience.com/causal-inference-with-synthetic-control-in-python-4a79ee636325
        https://github.com/OscarEngelbrektson/SyntheticControlMethods/blob/master/examples/user_guide.ipynb
        https://towardsdatascience.com/causal-inference-with-synthetic-control-using-python-and-sparsesc-9f1c58d906e6
        https://www.kaggle.com/code/aayushmnit/synthetic-control-using-python-and-sparsesc
        https://github.com/microsoft/SparseSC
        """
        return