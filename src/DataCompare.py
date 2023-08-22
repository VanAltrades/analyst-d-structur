class DataCompare(Data):
    """
    An extension of the Data class that allows us to compare a pre and post period.
    """

    def __init__(
            self,
            credential_path,
            client,
            sql_path,
            project,
            dataset,
            table,
            kpis,
            aggs,
            date_start,
            date_end,
            dim_date,
            dim_dimensions,
            where_clause,
            group_by_clause,
            date_change, # unique to this class and divides comparison periods into pre/post
            dimension_join_clause # unique to this subclass
    ):

        super().init__()
        self._dimension_join_clause = return_dynamic_pre_post_join_clause()

    def return_dynamic_pre_post_join_clause(DIM_LIST : list = dim_dimensions):
        """
        Returns a (JOIN) ON string that inserts pre/post dimension indexers from the DIM_LIST list.
        """
        OTHER_DIMENSION_JOIN = ""
        for i, dim in enumerate(DIM_LIST): 
            dim = dim.replace(",","")
            if i == 0: # first dimension join uses "ON"
                OTHER_DIMENSION_JOIN += f"ON pre.{dim}=post.{dim} "
            else: # other dimension joins use "AND"
                OTHER_DIMENSION_JOIN += f"AND pre.{dim}=post.{dim} " # add DIMENSION pre/post join string to final join
        return OTHER_DIMENSION_JOIN
    OTHER_DIMENSION_JOIN = return_dynamic_pre_post_join_clause()

    def get_change():
        return
    
    def get_new():
        return
    def get_lost():
        return
    def get_winners():
        return
    def get_losers():
        return
    
    def show_distribution():
        return
    def get_timelapse():
        return
    def get_scatter():
        return