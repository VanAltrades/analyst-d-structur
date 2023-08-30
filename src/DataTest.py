from .Data import Data
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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

#  ██████╗ █████╗ ██╗   ██╗███████╗ █████╗ ██╗         ██╗███╗   ███╗██████╗  █████╗  ██████╗████████╗
# ██╔════╝██╔══██╗██║   ██║██╔════╝██╔══██╗██║         ██║████╗ ████║██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
# ██║     ███████║██║   ██║███████╗███████║██║         ██║██╔████╔██║██████╔╝███████║██║        ██║   
# ██║     ██╔══██║██║   ██║╚════██║██╔══██║██║         ██║██║╚██╔╝██║██╔═══╝ ██╔══██║██║        ██║   
# ╚██████╗██║  ██║╚██████╔╝███████║██║  ██║███████╗    ██║██║ ╚═╝ ██║██║     ██║  ██║╚██████╗   ██║   
#  ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝    ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═╝  

    def get_causal_impact(self, test_group="Test", metric_field="visits", agg="sum"):
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
        # Convert 'date' column to datetime type for comparison
        self._data_sql[self._dim_sql_date] = pd.to_datetime(self._data_sql[self._dim_sql_date])

        pre_df = self._data_sql[self._data_sql[self._dim_sql_date] <= self._date_test_prior]
        post_df = self._data_sql[self._data_sql[self._dim_sql_date] >= self._date_test]

        self._data_pre = pre_df
        self._data_post = post_df

    def get_data_pre_post_comparison(self, dimension_column: str, metric_column : str, metric_column_agg : str = 'sum'):
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

    @staticmethod
    def get_new_dimensions(df_pre_post_comparison, metric):
        return df_pre_post_comparison[(df_pre_post_comparison[f'{metric}_post'].notnull()==True) & (df_pre_post_comparison[f'{metric}_pre'].isnull()==False)]

    @staticmethod
    def get_lost_dimensions(df_pre_post_comparison, metric):
        return df_pre_post_comparison[(df_pre_post_comparison[f'{metric}_pre'].notnull()==True) & (df_pre_post_comparison[f'{metric}_post'].isnull()==True)]

    @staticmethod
    def get_improving_dimensions(df_pre_post_comparison):
        return df_pre_post_comparison[df_pre_post_comparison[f'delta']>0]

    @staticmethod
    def get_declining_dimensions(df_pre_post_comparison):
        return df_pre_post_comparison[df_pre_post_comparison[f'delta']<=0]



def get_outliers_by_metrics():
    return

def get_outliers_by_dimensions():
    return

def get_test_group():
    return

def get_control_group():
    return

def get_length_estimate():
    return

def get_diff_in_diff():
    return

# ███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗███████╗████████╗██╗ ██████╗     ██████╗ ██████╗ ███╗   ██╗████████╗██████╗  ██████╗ ██╗     
# ██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║██╔════╝╚══██╔══╝██║██╔════╝    ██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔══██╗██╔═══██╗██║     
# ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║█████╗     ██║   ██║██║         ██║     ██║   ██║██╔██╗ ██║   ██║   ██████╔╝██║   ██║██║     
# ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║██╔══╝     ██║   ██║██║         ██║     ██║   ██║██║╚██╗██║   ██║   ██╔══██╗██║   ██║██║     
# ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║███████╗   ██║   ██║╚██████╗    ╚██████╗╚██████╔╝██║ ╚████║   ██║   ██║  ██║╚██████╔╝███████╗
# ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝     ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
                                                                                                                                         

def format_synthetic_control_df(self,col_metric="visits",col_breakout="breakout"):
    # use sql_path="../src/sql/data_test_synthetic_control.sql"
    df = self._data_sql.pivot(index=col_breakout, columns=self._dim_sql_date, values=col_metric)
    df.fillna(0,inplace=True)
    return df

def get_sythethic_control_object(self,test_id="Test",col_metric="visits",col_breakout="breakout"):
    df = format_synthetic_control_df(self,col_metric,col_breakout)
    sc_new = SparseSC.fit_fast( 
        features=df.iloc[:,df.columns <= self._date_test].values,
        targets=df.iloc[:,df.columns > self._date_test].values,
        treated_units=[idx for idx, val in enumerate(df.index.values) if val == test_id]
    )
    return sc_new

def plot_synthetic_control_gap(self, synth_df, test_id="Test"):
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
                'text':"Gap in Test v. Synthetic Control Visits",
                'y':0.95,
                'x':0.5,
            },
            legend =  dict(y=1, x= 0.8, orientation='v'),
            legend_title = "_legend_title_",
            xaxis_title="Date", 
            yaxis_title="Visits Trend",
            font = dict(size=15)
    )
    fig.show(renderer='notebook')
    # return

def get_synthetic_control_time_series_df(self, synth_df, test_id="Test"):
    ## creating required features
    features = synth_df.iloc[:,synth_df.columns <= self._date_test].values
    targets = synth_df.iloc[:,synth_df.columns > self._date_test].values
    treated_units = [idx for idx, val in enumerate(synth_df.index.values) if val == test_id] # [2]

    ## Fit fast model for fitting Synthetic controls
    sc_model = SparseSC.fit_fast( 
        features=features,
        targets=targets,
        treated_units=treated_units
    )

    result = synth_df.loc[synth_df.index == test_id].T.reset_index(drop=False)
    result.columns = ["date", "Observed"] 
    result['Synthetic_Control'] = sc_model.predict(synth_df.values)[treated_units,:][0]
    result['Observed'] = result['Observed'].astype(int)
    result['Synthetic_Control'] = result['Synthetic_Control'].astype(int)
    return result
    

def plot_synthetic_control_assessment(self, result, col_metric="visits"):
    fig = px.line(
            data_frame = result, 
            x = "date", 
            y = ["Observed","Synthetic_Control"], 
            template = "plotly_dark",)

    fig.add_trace(
        pgo.Scatter(
            x=[self._date_test,self._date_test],
            y=[result.Observed.min()*0.98,result.Observed.max()*1.02], 
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
            xaxis_title="Date", 
            yaxis_title=f"{col_metric}",
            font = dict(size=15)
    )
    fig.show(renderer='notebook')
    # return

def plot_synthetic_control_difference_across_time(self, result):
    #| code-fold: true
    #| fig-cap: Fig - Gap in Per-capita cigarette sales in California w.r.t Synthetic Control

    result['Test Effect'] = result['Observed'] - result['Synthetic_Control']
    fig = px.line(
            data_frame = result, 
            x = "date", 
            y = "Test Effect", 
            template = "plotly_dark",)
    fig.add_hline(0)
    fig.add_trace(
        pgo.Scatter(
            x=[self._date_test,self._date_test],
            y=[result["Test Effect"].min()*0.98,result["Test Effect"].max()*1.02], 
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
            legend_title = "_legend_title_2_",
            xaxis_title="Date", 
            yaxis_title="Gap in Visits",
            font = dict(size=15)
    )
    fig.show(renderer='notebook')
    # return

def get_synthetic_control_treatment_effect(self, result, col_metric="visits"):
    print(f"Effect of Change Event w.r.t Synthetic Control => {np.round(result.loc[result[self._dim_sql_date]==self._date_end,'Test Effect'].values[0],1)} {col_metric}")
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