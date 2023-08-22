# ██╗      ██████╗  ██████╗ ██╗ ██████╗    ██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗ 
# ██║     ██╔═══██╗██╔════╝ ██║██╔════╝    ██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗
# ██║     ██║   ██║██║  ███╗██║██║         ███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝
# ██║     ██║   ██║██║   ██║██║██║         ██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗
# ███████╗╚██████╔╝╚██████╔╝██║╚██████╗    ██║  ██║███████╗███████╗██║     ███████╗██║  ██║
# ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝ ╚═════╝    ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝

def print_sql_variable_logic(self):
    """
    Print all strings with {} in a set from string. 
    Instantiated to help with formatting print_sql_formatted()get_sql_data().

    Args:
        self._sql_variables.

    Returns:
        A set of strings with '{}' found in self._sql_file_string.
    """
    if len(self._sql_variables)>0:
        # print a set of the matches.
        print(f"""
            The following variables exist within {self._sql_path}:

            {self._sql_variables}
            
            Make sure the intented variable names in {self._sql_path} 
            exact match this class' respective variables, which include:
            
            {vars(self).keys()}

            The following sql will be updated:

            {self._sql_file_string}
            """)
    else:
        "Your SQL has no variables in it. get_sql_data() class method should work, query away!"

def print_sql_formatted(self):
    print(f"""
            This Class has formatted {self._sql_path} with the following variables:

            {self._sql_variables}

            The query now looks like:
            
            {self._sql_file_string_formatted}

            Run Data.get_sql_data() to return a Pandas.DataFrame from this script.
            """
            )