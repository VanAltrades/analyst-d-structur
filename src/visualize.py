import plotly.express as px
import numpy as np

# DataCompare

def get_distribution_pre_post(self, metric,**kwargs):
    fig = px.histogram(
        data_frame=self._data_pre_post_comparison, 
        x=[f"{metric}_pre", f"{metric}_post"],
        marginal="box", 
        color_discrete_map={f"{metric}_pre": "gray", f"{metric}_post": "blue"},
        barmode="overlay",
        **kwargs)
    fig.update_layout(title=f"Distribution of {metric.title()} (Pre and Post)")
    return fig.show()

def get_outlier_scatter(self, outliers):
    # Create a scatter plot with outliers
    fig = px.scatter(self._data_pre_post_comparison, x=self._data_pre_post_comparison.index, y='delta', color_discrete_sequence=['blue'], opacity=0.5, labels={'delta': 'Data'}, title=f'Scatterplot of Delta by {self._data_pre_post_comparison.index.name}')
    fig.add_trace(px.scatter(outliers, x=outliers.index, y='delta', color_discrete_sequence=['red'], labels={'delta': 'Outliers'}).data[0])

    # Customize the layout
    fig.update_traces(marker=dict(size=5))  # Adjust marker size
    fig.update_xaxes(tickangle=75)  # Rotate x-axis labels
    fig.update_layout(xaxis_title=f'{self._data_pre_post_comparison.index.name}', yaxis_title='Delta')
    fig.show()