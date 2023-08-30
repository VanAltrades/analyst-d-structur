import pandas as pd
import re
from .config import CREDENTIAL_PATH, CLIENT

#TODO: in this copy, I will move all class methods out of the __init__ and place them below. I will see if this corrects variables.
#     self.dim_value_list = self.get_dim_value_list()  # Call the method to initialize the list

#     def get_dim_value_list(self): # place this under class methods


class Data:
    """
    An abstract Data Structure based on time-series data.
    """
    from .logic import print_sql_variable_logic, print_sql_formatted
    from .trend import format_data_for_plotly, get_trend_line, get_trend_bar

    def __init__(self,sql_path,project,dataset,table,kpis,aggs,date_start,date_end,dim_sql_date,dim_sql_index,dim_sql_dimensions,where_clause):
        """
        Create a new Data instance.

        Data based on chosen sql script accessing a BigQuery Table.

        credential_path:        Path to GCP Service Account json file.
        client:                 GCP client.
        sql_path:               Path to SQL file to build data.
        project:                BigQuery project name as a string.
        dataset:                BigQuery dataset name as a string.
        table:                  BigQuery table name as a string.
        kpis:                   List of strings representing column names of metrics used from table.
        aggs:                   List of strings representing BigQuery aggregation functions for column names of metrics in table.
        date_start:             String representing the data's start date in the format YYYY-MM-DD.
        date_end:               String representing the data's end date in the format YYYY-MM-DD.
        dim_sql_date:               String representing the date column name to return within the table instantiated
        dim_sql_index:              String representing the index column name to return within the table instantiated
        dim_sql_dimensions:         List of strings representing column names of dimension used from table.
        where_clause:           String to include in the where clause i.e. "WHERE n LIKE x"
        group_by_clause:        dynamically generated - returns a GROUP BY clause with the number of _dim dimensions instantiated
        dimension_join_clause:  None
        """
        
        self._credential_path = CREDENTIAL_PATH
        self._client = CLIENT
        self._sql_path = sql_path
        self._project = project
        self._dataset = dataset
        self._table = table
        self._sql_file_string = self.get_sql_file_as_string()
        self._sql_variables = self.return_sql_variables()
        self._date_start = date_start
        self._date_end = date_end
        self._dim_sql_date = dim_sql_date
        self._dim_sql_index = dim_sql_index
        self._dim_sql_dimensions = dim_sql_dimensions
        self._dim_dict = self.get_dim_item_dict()
        self._dim_key_list = self.get_dim_key_list()
        self._dim_value_list = self.get_dim_value_list()
        self._kpis = kpis
        self._aggs = aggs
        self._kpi_aggregates = self.get_kpi_aggregate_functions()
        self._where_clause = where_clause
        self._group_by_clause = self.get_dynamic_group_by_clause()
        self._sql_file_string_formatted = self.get_sql_formatted()
        vars


    # ██████╗  █████╗ ████████╗ █████╗     ██╗███╗   ██╗███████╗████████╗ █████╗ ███╗   ██╗ ██████╗███████╗    ███╗   ███╗███████╗████████╗██╗  ██╗ ██████╗ ██████╗ ███████╗
    # ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝██╔════╝    ████╗ ████║██╔════╝╚══██╔══╝██║  ██║██╔═══██╗██╔══██╗██╔════╝
    # ██║  ██║███████║   ██║   ███████║    ██║██╔██╗ ██║███████╗   ██║   ███████║██╔██╗ ██║██║     █████╗      ██╔████╔██║█████╗     ██║   ███████║██║   ██║██║  ██║███████╗
    # ██║  ██║██╔══██║   ██║   ██╔══██║    ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║╚██╗██║██║     ██╔══╝      ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║   ██║██║  ██║╚════██║
    # ██████╔╝██║  ██║   ██║   ██║  ██║    ██║██║ ╚████║███████║   ██║   ██║  ██║██║ ╚████║╚██████╗███████╗    ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║╚██████╔╝██████╔╝███████║
    # ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝    ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝
                                                                                                                                                                        

    def get_sql_file_as_string(self):
        fd = open(self._sql_path, 'r')
        sql_file_string = fd.read()
        fd.close()
        return sql_file_string
    
    def return_sql_variables(self):
        """Returns all strings betweeen {} as a set from string self._sql_file_string.

        Args:
            self._sql_file_string.

        Returns:
            A set of strings with found in self._sql_file_string.
        """
        # Create a regular expression to match strings with {}.
        pattern = re.compile(r'\{([^\}]+)\}')
        # Find all matches for the regular expression.
        matches = pattern.findall(self._sql_file_string)
        return set(matches)

    def get_dim_item_dict(self):
        """
        Returns a filtered dictionary of the instance's existing var()s where the keys contain "_dim_sql".
        We only care about the variables with a prefix of "_dim_sql" because this Class's naming convention starts SQL dimensions with "_dim_sql".
        We require these dimensions to dynamically insert these dimensions into the SQL with commas in the right place and GROUP BY automatically generated.

        Args:
            vars(self) : dict : existing class instance variables in a dictionary before this function is called.

        Returns:
            res : dict : a dictionary of variables whose keys contain "_dim_sql" 
        """
        search_key = "_dim_sql_"
        d = vars(self)            
        res = dict(filter(lambda item: search_key in item[0], d.items()))
        return res
    
    def get_dim_key_list(self):
        """
        Returns existing class instance variable keys containing "_dim_sql" as strings in a list

        Args:            
            self._dim_dict : dict

        Returns:
            list : ex. ['_dim_sql_date', '_dim_sql_index', '_dim_sql_dimensions']
        """
        return list(self._dim_dict.keys())
    
    def get_dim_value_list(self):
        """
        Returns existing class instance variable values containing "_dim_sql" as strings in a list.
        Contains logic to handle different dict value data structures and formats them as strings in a list.

        Args:
            self._dim_dict : dict

        Returns:
            dim_value_list : list : ex. ['date', 'entry_page', 'pagetype', 'product_type']
        """
        dim_value_list = []
        for dim_values in self._dim_dict.values():            
            if isinstance(dim_values,str):
                dim_value_list += dim_values.split()
            elif isinstance(dim_values,list):
                for d in dim_values:
                    dim_value_list += d.split()
            else: # dimension is None
                continue
        return dim_value_list

    def get_kpi_aggregate_functions(self):
        """
        Returns a string the inserts into a sql file. The string applies aggredate functions to kpis. ex: "sum(visits), sum(revenue),"
        
        Args:
            self._kpis
            self._aggs
        
        Returns:
            kpi_aggs
        """
        kpi_aggs = ""
        for kpi, agg in zip(self._kpis,self._aggs):
            kpi_aggs += f"{agg}({kpi}) {kpi}, " # adds commas between and after metrics
        return kpi_aggs

    def get_dynamic_group_by_clause(self):
        """
        Returns a GROUP BY string that inserts the number of dimensions included based on date dim, index dim, and dimension dims strings converted to a _dim_value_list list.
        """
        GROUP_BY_CLAUSE = "GROUP BY "
        DIM_COUNT = len(self._dim_value_list) # length of dimensions + index
        for i in range(DIM_COUNT):
            if i == max(range(DIM_COUNT)): # if last iteration, don't include trailing comma
                n = str(i+1)
                GROUP_BY_CLAUSE = GROUP_BY_CLAUSE + f"{n}"
            else: # add indexer to group by clause with trailing comma
                n = str(i+1)
                GROUP_BY_CLAUSE = GROUP_BY_CLAUSE + f"{n},"
        return GROUP_BY_CLAUSE

    def get_sql_formatted(self):
        """
        Returns a sql string with {variables} replaced with exact matched class variable names' values.
        Please note that variables in .sql files should exactly match the names of the class variables.
        ex: {_project} in .sql will be replaced by Data's "_project" value of "e-comm-project",
        but {project} in .sql will not be replaced because it has no exact match class name... should be named {_project}
        
        Args:
            vars(self)
            self._sql_file_string
            self._dim_key_list

        Returns:
            sql_string with Class variables replacing templated {_variables} with exact match names
        """
        # dictionary of local class attributes and values
        dict = vars(self)
        # the unformatted sql string
        sql_string = self._sql_file_string
        # loop through the class' local __init__ attributes and replace them in the sql if their strings exact match
        for key, value in dict.items():
            # class variable name matches string in sql
            if sql_string.find(key) != -1:
                # sql_string = sql_string.replace(key, value)

                # value in dim key list and needs commas in SQL syntax
                if isinstance(value,str) and key in self._dim_key_list:
                    sql_string = sql_string.replace(key, value + ",") # replace sql variable with instance variable value
                # value not in dim list and does not need commas in SQL syntax (date_start,date_end,group_by_clause)
                elif isinstance(value,str):
                    sql_string = sql_string.replace(key, value) # replace sql variable with instance variable value
                elif isinstance(value,list):
                    value = ",".join(value)+"," # add commas between list values for SQL syntax
                    sql_string = sql_string.replace(key, value) # replace sql variable with instance variable value
                elif value is None:
                    sql_string = sql_string.replace(key, "") # replace sql variable with "" if instance variable has None value

        # replace the "{" and "}" strings in the sql. these were originally intended as f-strings, but f-strings required a manual .format()
        sql_string = sql_string.replace("{","").replace("}","")
        sql_string = sql_string.replace(",,",",")
        return sql_string
    

    def get_data_sql(self, sql_custom=None):
        """
        Query Google BigQuery and return a Pandas DataFrame object

        Args:
            self._client : BigQuery client from Class instance
            self._sql_file_string_formatted : string
                sql code string to send to BigQuery
            sql_custom : str
                If None then use dynamic SQL insertion provided by class instance variables, else provide custom SQL str as an argument

        Returns:
            self._data_sql : pd.DataFrame
                pandas dataframe object from the sql response
        
        """
        if sql_custom is None:
            query_job=self._client.query(self._sql_file_string_formatted) 
            results = query_job.result()
            df = results.to_dataframe()
            df
            self._data_sql = df
        else:
            query_job=self._client.query(sql_custom) 
            results = query_job.result()
            df = results.to_dataframe()
            df
            self._data_sql = df            