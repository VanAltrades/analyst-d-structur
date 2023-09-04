with pre as (
  SELECT


  DISTINCT {_dim_index}
  {_date_start_comparison} as date,
  {_dim_dimensions}
  {_kpi_aggregates}

  FROM `{_project}.{_dataset}.{_table}` facts
  WHERE 
  date BETWEEN "{_date_start_comparison}" AND "{_date_end_comparison}"
  {_where_clause}
  {_group_by_clause}
),

post as (
  DISTINCT {_dim_index}
  "Post" as period,
  {_date_start} as date,
  {_dim_dimensions}
  {_kpi_aggregates}

  FROM `{_project}.{_dataset}.{_table}` facts
  WHERE 
  date BETWEEN "{_date_start}" AND "{_date_end}"
  {_where_clause}
  {_group_by_clause}
)

select 
*
-- {_period} as span,
from pre
FULL OUTER JOIN post
  on pre.{_dim_index} = post.{_dim_index}