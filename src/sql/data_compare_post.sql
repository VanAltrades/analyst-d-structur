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