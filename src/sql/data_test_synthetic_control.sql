with test as (
  SELECT
  distinct {_dim_sql_date}
  "Test" as breakout,
  {_kpi_aggregates}
  FROM 
    `{_project}.{_dataset}.{_table}` facts
  INNER JOIN
    `{_batch_tbl}` batch
  on 
    REGEXP_EXTRACT(facts.{_dim_sql_index},r"{_regex}") = REGEXP_EXTRACT(batch.{_dim_test_index},r"{_regex}")    
  WHERE
    date BETWEEN "{_date_start}" AND "{_date_end}"
    {_where_clause}
    GROUP BY 1,2 -- {_group_by_clause} -- includes _index from join so manual for now
),
non_test as (
  SELECT
  distinct {_dim_sql_date},
  product_type as breakout, -- {_dim_sql_dimensions} as breakout, -- inserts "," after variable 
  {_kpi_aggregates}
  FROM
    `{_project}.{_dataset}.{_table}` facts
  LEFT OUTER JOIN
    `{_batch_tbl}` batch
  ON
    REGEXP_EXTRACT(facts.{_dim_sql_index},r"{_regex}") = REGEXP_EXTRACT(batch.{_dim_test_index},r"{_regex}")    
  WHERE
    date BETWEEN "{_date_start}" AND "{_date_end}"
    -- and batch.test_group <> "Test"
    {_where_clause}
    GROUP BY 1,2 -- {_group_by_clause} -- includes _index from join so manual for now
)
SELECT *
FROM test
UNION ALL 
SELECT *
FROM non_test
ORDER BY 1 DESC