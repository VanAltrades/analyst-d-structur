from .Data import Data
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy import stats

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

    def get_historic_metric_mean(self, metric):
        # print total, show trend
        df = self._data_sql
        
        # pre period test data
        date_test_prior = datetime.strptime(self._date_test_prior, "%Y-%m-%d")
        # Convert the 'self._dim_sql_date' column in the DataFrame to datetime format
        df[self._dim_sql_date] = pd.to_datetime(df[self._dim_sql_date])
        df = df.loc[(df["test_group"]=="Test")&(df[self._dim_sql_date]<=date_test_prior)]
        
        historic_mean = df[metric].mean() 
        print(f"historic mean of {metric}: {historic_mean}")

    def get_historic_metric_std(self, metric):
        # print total, show trend
        df = self._data_sql
        
        # pre period test data
        date_test_prior = datetime.strptime(self._date_test_prior, "%Y-%m-%d")
        # Convert the 'self._dim_sql_date' column in the DataFrame to datetime format
        df[self._dim_sql_date] = pd.to_datetime(df[self._dim_sql_date])
        df = df.loc[(df["test_group"]=="Test")&(df[self._dim_sql_date]<=date_test_prior)]
        
        historic_std = df[metric].std() # variability
        print(f"historic std. dev. of {metric}: {historic_std}")

        return

    def estimate_time_to_significance(self, metric, post_change_mean, alpha=0.05, power=0.8, alternative='two-sided'):
        """
        Estimate the time (in terms of sample size) required to achieve statistical significance
        for a hypothesis test comparing a post-change mean to a historic mean.

        Parameters:
        - self: The instance of the class containing the data and configuration.
        - metric (str): The metric for which significance is being estimated.
        - post_change_mean (float): The expected post-change mean value.
        - alpha (float, optional): The significance level, typically set to 0.05.
        - power (float, optional): The desired statistical power, typically set to 0.8.

        Returns:
        - required_sample_size (float): The estimated sample size required to achieve significance.

        This function calculates the required sample size needed to detect a significant change in
        a given metric by performing a hypothesis test. It uses historic data to estimate the
        standard deviation, mean, and sample size, and then calculates the effect size. With the
        effect size, significance level (alpha), and desired power, it computes the required sample
        size using a power analysis formula. The result is the estimated number of samples or
        observations needed to detect a meaningful change in the metric.

        Note: The function assumes daily data collection and a one-sided test for an increase in
        the metric (alternative='larger').
        """
        self._data_sql[self._dim_sql_date] = pd.to_datetime(self._data_sql[self._dim_sql_date])
        self._date_test_prior = pd.to_datetime(self._date_test_prior)
        
        pre_df = self._data_sql[self._data_sql[self._dim_sql_date] <= self._date_test_prior]
        post_df = self._data_sql[self._data_sql[self._dim_sql_date] >= self._date_test]

        df = self._data_sql
        
        # pre period test data
        # date_test_prior = datetime.strptime(self._date_test_prior, "%Y-%m-%d")
        
        # Convert the 'self._dim_sql_date' column in the DataFrame to datetime format
        df[self._dim_sql_date] = pd.to_datetime(df[self._dim_sql_date])
        df = df.loc[(df["test_group"]=="Test")&(df[self._dim_sql_date]<=self._date_test_prior)]
        
        # Calculate standard deviation and sample size from historic data
        historic_std = df[metric].std() # variability
        print(f"historic std. dev. of {metric}: {historic_std}")
        historic_mean = df[metric].mean() 
        print(f"historic mean of {metric}: {historic_mean}")
        sample_size_historic = len(df)
        print(f"historic sample size {sample_size_historic}")

        # Calculate effect size
        effect_size = (historic_mean - post_change_mean) / historic_std

        # Calculate the required sample size (time) for the desired power and alpha
        required_sample_size = sm.stats.tt_solve_power(
            effect_size=effect_size,
            alpha=alpha,        # Significance level (e.g., 0.05 for 95% confidence)
            power=power,        # Desired power (e.g., 0.8 for 80% power)
            alternative=alternative,  # Use 'larger' for one-sided test (increase) or 'two-sided' for increase or decrease
            nobs=None,  # Number of observations (sample size) is the unknown
            # ratio=len(post_df) / len(pre_df)
            # ratio=1,  # Assume 1:1 allocation to treatment and control
            # alternative='two-sided'  # Use 'two-sided' for a two-sided test
        )
        print(f"required sample size: {required_sample_size}")

        # Estimate the required monitoring duration (in days)
        # Assuming you collect data daily
        days_required = np.ceil(required_sample_size)
        print(f"{days_required} days required")

        # # Calculate the required test duration in days
        # # Divide by daily sample size to get the number of days required
        # daily_sample_size = len(pre_df)
        # required_duration_days = np.ceil(required_sample_size / daily_sample_size) # By dividing the required_sample_size by the daily_sample_size, you can estimate how many days you should run your test to collect the necessary data to reach statistical significance. The result, required_duration_days, gives you an estimate of the test duration in days.

        # print(f"Estimated test duration in days: {required_duration_days} (when dividing required_sample_size by days in pre period)")

        return required_sample_size
    
    def ttest_significance_level(self, metric, alpha = 0.05):
        # must use daily df

        pre_df = self._data_sql[self._data_sql[self._dim_sql_date] <= self._date_test_prior]
        post_df = self._data_sql[self._data_sql[self._dim_sql_date] >= self._date_test]

        pre_test_mean_metric = pre_df[f'{metric}'].mean()
        pre_test_std_metric = pre_df[f'{metric}'].std()
        post_test_mean_metric = post_df[f'{metric}'].mean()
        post_test_std_metric = post_df[f'{metric}'].std()

        # Perform a t-test for metric
        t_stat_metric, p_value_metric = stats.ttest_ind_from_stats(
            pre_test_mean_metric, pre_test_std_metric, len(pre_df),
            post_test_mean_metric, post_test_std_metric, len(post_df)
        )

        # Check if the p-values are less than the significance level
        if p_value_metric < alpha:
            print("Post-period data is statistically different. Stop the test.")
        else:
            print("Continue the test.")


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

    def get_data_pre_post_comparison(self, dimension_column: str, metric_column : str, metric_column_agg : str = 'sum', per_day=False):
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
        result_df = pre_grouped_by_dimension.merge(post_grouped_by_dimension, how='outer', left_index=True, right_index=True, suffixes=('_pre', '_post'))

        if per_day is False:
            pass
        else:
            result_df[f'{metric_column}_pre'] = (result_df[f'{metric_column}_pre']/self._days_in_pre.days)
            result_df[f'{metric_column}_post'] = (result_df[f'{metric_column}_post']/self._days_in_post.days)

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
        Format the input DataFrame for synthetic control analysis.

        This function takes an input DataFrame containing historical data with specific columns
        for date, metric, and breakout information. It prepares the DataFrame for synthetic
        control analysis by performing the following steps:

        1. Converts the date column to a datetime format to ensure proper date handling.
        2. Reshapes the DataFrame into a pivot table with breakout values as rows, dates as
        columns, and metric values as the data.
        3. Fills missing values with zeros, ensuring that the resulting DataFrame is suitable
        for the synthetic control analysis.

        Parameters:
            - col_date (str): Name of the column containing date information.
            - col_metric (str): Name of the column containing metric data.
            - col_breakout (str): Name of the column containing breakout information.

        Returns:
            - df (pd.DataFrame): A formatted DataFrame suitable for synthetic control analysis.
            It has breakout values as rows, dates as columns, and metric values as data.

        Example Usage:
            formatted_data = format_synthetic_control_df(df, col_date="date", col_metric="visits", col_breakout="breakout")
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
        Generate and display a plot illustrating the gap between the test group and the synthetic control
        in terms of a specified metric over time.

        This function takes a formatted synthetic control DataFrame, identifies the treated unit based on the
        specified test ID, and calculates the synthetic control values for the treated unit and the metric
        specified. It then creates a plot to visualize the time series of the test group's metric values
        alongside the synthetic control's metric values, emphasizing any gaps or changes.

        Parameters:
            - col_date (str): Name of the date column in the input DataFrame.
            - col_metric (str): Name of the metric column in the input DataFrame.
            - col_breakout (str): Name of the breakout column in the input DataFrame.
            - test_id (str): Identifier for the unit under test in the input DataFrame.

        Returns:
            - None: The function displays the plot but does not return a value.

        Note:
        The function relies on internal attributes like _dim_sql_date and _date_test, which are not explicitly
        defined within the function but are expected to be set elsewhere in the class or object.

        Example Usage:
        To visualize the gap in "visits" metric between the "Test" group and its synthetic control over time,
        you can call the function as follows:
        obj.plot_synthetic_control_gap(col_metric="visits", test_id="Test")
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
                    'text':f"Gap in Test v. Control {col_metric.title()} | Gap between the test group and control group in terms of {col_metric.title()} over time",
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

    def get_synthetic_control_time_series_df(self, fit_fast=True,col_date="date",col_metric="visits",col_breakout="breakout", test_id="Test"):
        """
        Generate a time series DataFrame for synthetic control analysis.

        This function takes a DataFrame with time series data and generates a time series DataFrame
        for synthetic control analysis. It assumes that the input DataFrame has a specified date
        column, a metric column (e.g., visits), and a breakout column to identify different units.
        The function performs the following steps:

        1. Formats the input DataFrame to ensure it is suitable for synthetic control analysis.
        2. Separates the data into features and targets based on a specified test date.
            The choice of using the data prior to the change date as "features" and the data after the change date as "targets" in the synthetic control analysis serves a specific purpose in this context. This approach is designed to create a meaningful basis for comparison and estimation of treatment effects. Here's why this division is commonly made:
            Causal Inference:
                The primary goal of synthetic control analysis is to estimate the causal effect of a treatment or intervention applied to a specific unit (the treated unit). You want to answer the question: "What would have happened to the treated unit if it had not received the treatment?"
                To answer this question, you need to establish a counterfactual scenario. That is, you need to construct a synthetic version of the treated unit's post-treatment behavior based on how similar units behaved before and after their own "change date" (which could be a policy change, intervention, event, etc.).
            Creating a Counterfactual:
                By using data from similar units (features) before the change date as "features," you are essentially constructing a historical behavior pattern of the treated unit under the assumption that it was not treated. This is your counterfactual scenario.
                The "targets" consist of the treated unit's actual data after the change date, which represents the observed outcome after the treatment.
            Estimating Treatment Effects:
                After fitting the synthetic control model, you generate a synthetic control time series for the treated unit using the features and learned model weights. This synthetic control time series represents what would have happened to the treated unit's metric if it had not been treated.
                By comparing the synthetic control time series (counterfactual) with the observed data of the treated unit (the actual outcome), you can estimate the treatment effect. The difference between the two, often referred to as the "Test Effect," quantifies the impact of the treatment on the treated unit.
            In summary, separating the data into features (pre-change date) and targets (post-change date) is a critical step in the synthetic control analysis. It allows you to construct a counterfactual scenario and estimate the causal effect of the treatment by comparing what actually happened to what would have happened in the absence of the treatment.
        3. Checks if the number of features and targets have the same number of rows.
        4. Identifies the treated unit (unit under test) based on a specified test ID.
        5. Fits a fast synthetic control model to the features and targets.
        6. Creates a time series DataFrame containing observed and synthetic control values
            along with the test effect (the difference between observed and synthetic control).
            
        Parameters:
            - col_date (str): Name of the date column in the input DataFrame.
            - col_metric (str): Name of the metric column in the input DataFrame.
            - col_breakout (str): Name of the breakout column in the input DataFrame.
            - test_id (str): Identifier for the unit under test in the input DataFrame.

        Returns:
            - synth_df_timeseries (pd.DataFrame): A time series DataFrame containing columns
                'date' (date values), 'Observed' (observed metric values), 'Synthetic_Control'
                (synthetic control metric values), and 'Test Effect' (the difference between
                observed and synthetic control).

        Raises:
            - ValueError: If there is an issue with model fitting or if features and targets
                have different numbers of rows.
        """

        synth_df = self.format_synthetic_control_df(col_date,col_metric,col_breakout)
        
        date_array_as_datetime = pd.to_datetime(synth_df.columns)
        try:
            date_test = datetime.strptime(self._date_test, '%Y-%m-%d')
        except:
            date_test = datetime.datetime.strptime(self._date_test, '%Y-%m-%d')

        ## creating required features
        features = synth_df.iloc[:,date_array_as_datetime <= date_test].values # pre
        targets = synth_df.iloc[:,date_array_as_datetime > date_test].values # post

        # Check if features and targets have the same number of rows
        if features.shape[0] != targets.shape[0]:
            raise ValueError("Features and targets must have the same number of rows.")
        
        treated_units = [idx for idx, val in enumerate(synth_df.index.values) if val == test_id] # [2]

        ## Fit fast model for fitting Synthetic controls
        try:
            # Fit the model:
            # During training, the model tries to find the right combination of weights for the features 
            # so that when you multiply each feature's data by its weight and sum them up, 
            # you get a synthetic series that closely matches the treated unit's historical data.
            if fit_fast is True:
                sc_model = SparseSC.fit_fast(
                    features=features,
                    targets=targets,
                    treated_units=treated_units
                )
            else:
                # Split your data into a training set and a test set
                X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

                # Fit your model on the training set
                sc_model = SparseSC.fit(
                    features=X_train,
                    targets=y_train,
                    treated_units=treated_units
                )
                # Evaluate your model on the test set
                score = sc_model.score(X_test, y_test)

                # sc_model = SparseSC.fit(
                #     features=features,
                #     targets=targets,
                #     treated_units=treated_units,
                #     n_splits=5  # Use 5-fold cross-validation... update to the number of splits in your example
                # )
        except Exception as e:
            raise ValueError("Error occurred during model fitting: {}".format(str(e)))

        # create a time series DataFrame for the treated unit.
        synth_df_timeseries = synth_df.loc[synth_df.index == test_id].T.reset_index(drop=False)
        synth_df_timeseries.columns = ["date", "Observed"] 
        # Calculate the Synthetic Control value for the treated unit at each time point.
        # Here, we use the trained synthetic control model (sc_model) to predict the synthetic
        # control values for the treated unit. The result is a time series of synthetic control
        # values. [treated_units,:] is used to select the row(s) corresponding to the treated unit,
        # and [0] is used to extract the first (and presumably only) row of synthetic control values
        # for the treated unit and assign it to the 'Synthetic_Control' column in synth_df_timeseries.
        synth_df_timeseries['Synthetic_Control'] = sc_model.predict(synth_df.values)[treated_units,:][0]
        synth_df_timeseries['Observed'] = synth_df_timeseries['Observed'].astype(int)
        synth_df_timeseries['Synthetic_Control'] = synth_df_timeseries['Synthetic_Control'].astype(int)
        synth_df_timeseries['Test Effect'] = synth_df_timeseries['Observed'] - synth_df_timeseries['Synthetic_Control']
        return synth_df_timeseries        

    def plot_synthetic_control_assessment(self, synth_df_timeseries, col_date="date",col_metric="visits"):
        """
        Plot the assessment of the synthetic control model's performance.

        This function generates a line plot to visually assess how well the synthetic control
        model replicates the observed data. It takes as input a DataFrame (`synth_df_timeseries`)
        containing time series data, where columns 'date,' 'Observed,' and 'Synthetic_Control'
        represent the date, observed metric values, and synthetic control metric values,
        respectively.

        Parameters:
            - synth_df_timeseries (pd.DataFrame): A DataFrame containing time series data for
            observed and synthetic control metrics.
            - col_date (str): Name of the date column in the DataFrame (default: "date").
            - col_metric (str): Name of the metric column in the DataFrame (default: "visits").

        Returns:
            - None

        This function generates a line plot using Plotly (px.line) to visualize the observed and
        synthetic control metrics over time. It also adds a vertical dashed line to indicate
        the date of a change event, typically representing the start of a treatment or
        intervention.

        The plot's title, axis labels, legend placement, and font size are customized for
        readability. The resulting plot is displayed in a Jupyter Notebook using the
        'notebook' renderer.
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
                    'text':f"Synthetic Control Assessment | {col_metric.title()} | How well the synthetic control model replicates the observed data",
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
                    'text':f"Difference across time | Observed {col_metric.title()} deviation from the synthetic control {col_metric.title()} over time",
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
        Calculate and summarize the treatment effect of a change event with respect to the synthetic control.

        This function takes a time series DataFrame generated from a synthetic control analysis
        (usually containing observed, synthetic control, and test effect columns), and calculates
        and summarizes the treatment effect of a change event with respect to the synthetic control.
        
        Parameters:
            - synth_df_timeseries (pd.DataFrame): Time series DataFrame containing columns
            'date' (date values), 'Observed' (observed metric values), 'Synthetic_Control'
            (synthetic control metric values), and 'Test Effect' (the difference between
            observed and synthetic control).
            - col_metric (str): Name of the metric for which the treatment effect is calculated.

        Returns:
            None

        Prints:
            - A summary of the treatment effect of the change event in terms of the specified metric.
            - The total lift in the specified metric after the change event.

        The function filters the DataFrame to include only rows after the change date and calculates
        the total lift, which represents the cumulative difference between the observed metric
        and the synthetic control metric after the change event.
        """
        print("Summary of the treatment effect of a change event with respect to the synthetic control")
        print(f"Impact of the intervention or treatment on {col_metric.title()}")
        # print(f"Effect of Change Event w.r.t Synthetic Control => {np.round(synth_df_timeseries.loc[synth_df_timeseries['date']==self._date_end,'Test Effect'].values[0],1)} {col_metric}")         
        
        # Filter the dataframe to include only rows after the change date
        df = synth_df_timeseries[synth_df_timeseries['date'] > self._date_test]
        # Calculate the total lift
        total_lift = df['Test Effect'].sum()
        print(f'Effect of Change Event w.r.t Synthetic Control => Total Lift after change: {total_lift} {col_metric}')
        # return

    def get_synthetic_control_diff_in_diff(self, synth_df_timeseries,col_metric="visits"):
        """
        Perform a Difference-in-Differences (DiD) analysis on Synthetic Control time series data.

        This function takes a time series DataFrame generated from a Synthetic Control analysis
        and performs a DiD analysis to assess the impact of a treatment or intervention on a
        specific metric. The DiD analysis involves comparing the behavior of the treated group
        (the unit under test) and the synthetic control group (similar units) before and after
        the treatment.

        Parameters:
            - synth_df_timeseries (pd.DataFrame): Time series DataFrame containing columns
            'date' (date values), 'Observed' (observed metric values), and 'Synthetic_Control'
            (synthetic control metric values).
            - col_metric (str): Name of the metric being analyzed (e.g., 'visits').

        Prints:
            - Difference-in-Differences (DiD) analysis results:
            - DiD of Average Metric: The change in the average metric value for the treated group
                compared to the synthetic control group before and after the treatment.
            - DiD of Total Metric per Day: The change in the total metric value per day for the
                treated group compared to the synthetic control group before and after the treatment.

        Note:
            - The function assumes that the Synthetic Control time series has been properly
            generated with the treated unit and synthetic control values.

        """
        # Split data into treatment and control groups
        treatment_group = synth_df_timeseries[['date','Observed']]
        control_group = synth_df_timeseries[['date','Synthetic_Control']]

        # Calculate means for treatment and control groups in pre and post periods
        treatment_pre_mean = treatment_group[(treatment_group['date']>=self._date_start)&(treatment_group['date']<=self._date_test_prior)]['Observed'].mean()
        treatment_post_mean = treatment_group[(treatment_group['date']>=self._date_test)&(treatment_group['date']<=self._date_end)]['Observed'].mean()
        control_pre_mean = control_group[(control_group['date']>=self._date_start)&(treatment_group['date']<=self._date_test_prior)]['Synthetic_Control'].mean()
        control_post_mean = control_group[(control_group['date']>=self._date_test)&(control_group['date']<=self._date_end)]['Synthetic_Control'].mean()

        # Synthetic Control Lift
        control_lift = control_post_mean - control_pre_mean
        print("\n",f"Synthetic Control Pre/Post Lift of Average {col_metric.title()}: {round(control_lift,2)}")
        # Test Lift 
        test_lift = treatment_post_mean - treatment_pre_mean
        print(f"Treatment Group Pre/Post Lift of Average {col_metric.title()}: {round(test_lift,2)}")

        # Calculate DiD for average of metrics
        did_avg = (treatment_post_mean - treatment_pre_mean) - (control_post_mean - control_pre_mean)

        print(f"Difference-in-Differences (Treated Group's Lift[pre/post] over Test Group) of Average {col_metric.title()}:", round(did_avg,2))

        # Calculate totals/# of days for treatment and control groups in pre and post periods
        treatment_pre_per_day = (treatment_group[(treatment_group['date']>=self._date_start)&(treatment_group['date']<=self._date_test_prior)]['Observed'].sum()/self._days_in_pre.days)
        treatment_post_per_day = (treatment_group[(treatment_group['date']>=self._date_test)&(treatment_group['date']<=self._date_end)]['Observed'].sum()/self._days_in_pre.days)
        control_pre_per_day = (control_group[(control_group['date']>=self._date_start)&(treatment_group['date']<=self._date_test_prior)]['Synthetic_Control'].sum()/self._days_in_pre.days)
        control_post_per_day = (control_group[(control_group['date']>=self._date_test)&(control_group['date']<=self._date_end)]['Synthetic_Control'].sum()/self._days_in_pre.days)
        
        # Synthetic Control Lift
        control_lift = control_post_per_day - control_pre_per_day
        print("\n",f"Synthetic Control Pre/Post Lift: {round(control_lift,2)} {col_metric.title()} per Day")
        # Test Lift 
        test_lift = treatment_post_per_day - treatment_pre_per_day
        print(f"Treatment Group Pre/Post Lift: {round(test_lift,2)} {col_metric.title()} per Day")

        # Calculate DiD for average of metrics
        did_per_day = (treatment_post_per_day - treatment_pre_per_day) - (control_post_per_day - control_pre_per_day)

        print(f"Difference-in-Differences (Treated Group's Lift[pre/post] over Test Group) of Total {col_metric.title()} per Day : {round(did_per_day,2)}" )





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