SELECT
    {_dim_sql_date}
    -- DISTINCT {_dim_index}
    {_dim_sql_index}
    {_dim_sql_dimensions}
    {_kpi_aggregates}

FROM 
    `{_project}.{_dataset}.{_table}` facts

WHERE 
    date BETWEEN "{_date_start}" AND "{_date_end}"
    {_where_clause}
    {_group_by_clause}