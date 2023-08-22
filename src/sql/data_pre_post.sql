with pre as (
  SELECT

  {_dim_date}
  DISTINCT {_dim_index}
  {_dim_dimensions}
  {_kpi_aggregates}

  FROM `{_project}.{_dataset}.{_table}` facts
  WHERE 
  date BETWEEN "{_date_start}" AND "{_date_end}"
  {_where_clause}
  {_group_by_clause}
),

post as (
  SELECT

  {DATE_DIMENSION}
  DISTINCT {INDEX}
  {DIMENSIONS}
  {POST_METRIC_AGGREGATIONS}

  FROM `{PROJECT_ID_STRING}.{DATASET_NAME_STRING}.{TABLE_NAME_STRING}` facts
  WHERE 
  date BETWEEN "{STRING_POST_START_DATE}" AND "{STRING_POST_END_DATE}"
  {WHERE_CLAUSE}
  {GROUP_BY_CLAUSE}
)

select *
from pre
FULL OUTER JOIN post
  {OTHER_DIMENSION_JOIN}