SELECT
*
    -- {_dim_sql_date}
    -- -- DISTINCT {_dim_index}
    -- {_dim_sql_index}
    -- {_dim_sql_dimensions}
    -- {_kpi_aggregates}
FROM `{_project}.{_dataset}.{_table}` facts
  INNER JOIN `{_batch_tbl}` batch
    on REGEXP_EXTRACT(facts.{_dim_sql_index},r"{_regex}") = REGEXP_EXTRACT(batch.{_dim_test_index},r"{_regex}")    
WHERE 
    date BETWEEN "{_date_start}" AND "{_date_end}"
    {_where_clause}
    -- {_group_by_clause}