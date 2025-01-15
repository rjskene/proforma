"""
Package for creating and managing proforma financial statements

TO DO:
* test to ensure that no account / subaccount has the same name
* there are ONLY TWO accounts: Assets and Capital (Liabilities and Equity)
* Net Income is a subset of Retained Earnings / Equity / Capital
"""
import typing
import warnings
import random
import string
import itertools as it
import datetime
import numpy as np
import pandas as pd
import networkx as nx

def networkx_to_treelib_string(G, max_depth=float('inf')):
    """
    Convert a NetworkX graph to a string representation using treelib
    """

    def build_tree_string(node, prefix="", is_last=False, is_root=False, depth=0, max_depth=2):
        output = []

        # Add current node
        if is_root:
            output.append(f"{node}")
        else:
            output.append(f"{prefix}{'└── '}{node}")

        # Get children (successors) of the current node
        if depth < max_depth:
            children = list(G.successors(node))

            # Recursively process children
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                new_prefix = prefix + "    "
                output.extend(build_tree_string(child, new_prefix, is_last=is_last_child, depth=depth+1, max_depth=max_depth))

        return output

    # Find all root nodes (nodes with no incoming edges)
    roots = [node for node in G.nodes() if G.in_degree(node) == 0]

    # Build the forest string
    forest = []
    for i, root in enumerate(roots):
        is_last_root = (i == len(roots) - 1)
        forest.extend(build_tree_string(root, is_last=is_last_root, is_root=True,  max_depth=max_depth))
        if not is_last_root:
            forest.append("")  # Add a blank line between root trees

    # Return the full forest string
    return "\n".join(forest)

BASE_ACCOUNTS = ('Assets', 'Capital')
class Chart(nx.DiGraph):        
    """
    A Chart of Accounts is a directed graph with nodes as accounts and edges as relationships between accounts
    """
    def __init__(self, *args, display_depth=2, **kwargs):
        kwargs['name'] = 'Chart of Accounts'
        super().__init__(self, *args, **kwargs)
        
        for account_name in self.BASE_ACCOUNTS:
            self.add_node(account_name, node_name=account_name.lower())

        self.add_subs('Capital', 'Liabilities', 'Equity')
        self.add_subs('Equity', 'Retained Earnings')
        self.add_subs('Retained Earnings', 'Income', 'Expenses')

        self.display_depth = display_depth
            
    def __getattr__(self, key):
        nodes = [x for x,y in self.nodes(data=True) if y['node_name']==key]
        if len(nodes) == 1:
            return nodes[0]
        else:
            return super().__getitem__(key)
    
    def __repr__(self):
        return self._tree_repr_()
    
    def __str__(self):
        return self._tree_repr_()
    
    @property
    def BASE_ACCOUNTS(self):
        """BASE ACCOUNTS PROPERTY IS IMMUTABLE"""
        return BASE_ACCOUNTS

    @property
    def accounts(self):
        return self.nodes

    def sub_accounts(self, *args, **kwargs):
        return nx.dfs_successors(self, *args, **kwargs)
    
    def _make_node_name(self, name):
        if not isinstance(name, str):
            raise ValueError('Sub-account names should be str')
            
        if ' ' in name:
            name = name.replace(' ', '_')
        return name.lower()
    
    def _tree_repr_(self):
        forest_string = networkx_to_treelib_string(self, max_depth=self.display_depth)

        return 'CHART OF ACCOUNTS\n\n' + forest_string
    
    # CRUD for Graph
    def add_sub(self, parent, sub, **kwargs):
        if parent not in self.find_accounts() and not self.in_degree(parent):
            raise ValueError(f'Parent account "{parent}" must be a base account or have a parent')
        
        if sub in self.nodes():
            raise ValueError(f'Node "{sub}" already exists in the chart')

        if 'node_name' not in kwargs:
            kwargs['node_name'] = self._make_node_name(sub) 

        self.add_node(sub, **kwargs)
        self.add_edge(parent, sub)
        
    def add_subs(self, parent, *subs, **kwargs):
        if parent not in self.find_accounts() and not self.in_degree(parent):
            raise ValueError(f'Parent account "{parent}" must be a base account or have a parent')

        existing_nodes = [sub for sub in subs if sub in self.nodes()]
        if existing_nodes:
            raise ValueError(f'Nodes {existing_nodes} already exist in the chart')

        subs_with_attrs = ((sub, {'node_name': self._make_node_name(sub)}) for sub in subs)
        edges = ((parent, sub) for sub in subs)

        self.add_nodes_from(subs_with_attrs)
        self.add_edges_from(edges)
    
    def change_parent(self, account:str, old_parent:str, new_parent:str):
        self.remove_edge(old_parent, account)
        self.add_edge(new_parent, account)
        return self

    def insert_between(self, old_parent:str, new_parent:str, subs:list[str]):
        self.add_node(new_parent)
        self.add_edge(old_parent, new_parent)
        self.add_edges_from([(n1, n2) for n1, n2 in it.product([new_parent], subs)])
        self.remove_edges_from([(n1, n2) for n1, n2 in it.product([old_parent], subs)])

        return self
    
    # Operators
    def find_terminal_subs(self):
        return [node for node in self.nodes() if self.in_degree(node) >= 1 and self.out_degree(node) == 0]

    def find_target_subs(self):
        return [node for node in self.nodes() if self.in_degree(node) >= 1]

    def find_accounts(self):
        return [n for n, d in self.in_degree() if d == 0]

    def accounts_in_order(self):
        return nx.dfs_postorder_nodes(self)

    def as_list(self):
        accounts = self.find_accounts()
        targets = accounts + self.find_target_subs() # Include top-level accounts as their own targets

        paths = []    
        for account, target in it.product(accounts, targets):
            try:
                path = nx.shortest_path(self, account, target)
                paths.append(path)
            except nx.NetworkXNoPath:
                pass

        return paths
    
    def as_frame(self):
        frame = pd.DataFrame(self.as_list()).fillna('---')
        frame = frame.sort_values(list(frame.columns)).reset_index(drop=True)
        return frame

    @classmethod
    def from_dict_of_lists(cls, dict_of_lists, *args, **kwargs):
        chart = cls(*args, **kwargs)
        graph = nx.from_dict_of_lists(dict_of_lists, create_using=nx.DiGraph)
        chart.add_nodes_from(graph.nodes(data=True))
        chart.add_edges_from(graph.edges())

        for node in chart.nodes:
            if 'node_name' not in chart.nodes[node]:
                chart.nodes[node]['node_name'] = chart._make_node_name(node)
        
        # if chart.find_accounts() != chart.BASE_ACCOUNTS:
        #     diff_base = set(list(chart.BASE_ACCOUNTS)) - set(chart.find_accounts())
        #     diff_chart = set(chart.find_accounts()) - set(list(chart.BASE_ACCOUNTS))
        #     raise ValueError(f'Chart must contain base accounts {diff_base} and not contain {diff_chart}')
        
        return chart
    
class Entry(dict):
    def __init__(self, account, date, amount=0, trans_id:str=None, **kwargs):
        date = self._format_date(date)

        super().__init__(account=account, date=date, amount=amount, trans_id=trans_id, **kwargs)

    def __neg__(self):
        self['amount'] *= -1
        return self

    def _format_date(self, date):
        if isinstance(date, pd.Timestamp):
            date = date.strftime('%Y-%m-%d')
        elif isinstance(date, datetime.date):
            date = date.isoformat()
        elif isinstance(date, datetime.datetime):
            date = date.strftime('%Y-%m-%d')
        elif isinstance(date, str):
            try:
                datetime.date.fromisoformat(date)
            except ValueError:
                raise ValueError("Incorrect data format, should be YYYY-MM-DD")
        
        return date

def transact(entries:typing.Union[Entry, list[Entry]], must_balance:bool=True):
    if isinstance(entries, Entry):
        entries = [entries]

    df = pd.DataFrame(entries)

    is_balanced = np.isclose(df.amount.sum(), 0, atol=1e-6)
    if not is_balanced:
        if must_balance: 
            raise ValueError('Entries do not sum to zero')
        else:
            warnings.warn('Entries do not sum to zero')

    df.date = pd.to_datetime(df.date)

    return df

class TransactionIDCounter:
    def __init__(self, prefix:str=None):
        self.counter = 1
        self.prefix = prefix if prefix else ''.join(random.choices(string.ascii_uppercase, k=4))

    def __str__(self):
        return self.value()
    
    def __repr__(self):
        return self.value()

    def value(self):
        return f'{self.prefix}{self.counter:012d}'

    def __call__(self):
        value = self.value()
        self.counter += 1
        return value

class Ledger:
    """
    A ledger is a collection of transactions.
    """
    def __init__(self, 
        chart, 
        transactions:typing.Union[typing.List[pd.DataFrame], pd.DataFrame]=[],
        counter_kwargs:dict={}
    ):
        if isinstance(transactions, pd.DataFrame):
            transactions = [transactions]

        self._id_counter = TransactionIDCounter(**counter_kwargs)

        self.chart = chart
        self.transactions = self._clean_transactions(transactions)
        self.statement = Statement(self)

    def __repr__(self):
        return self.transactions.__repr__()
    
    def __str__(self):
        return self.transactions.__str__()

    @property
    def stat(self):
        return self.statement

    def _clean_transactions(self, transactions):
        """
        Clean transactions:
        - generate trans_id if not provided
        - check that all accounts are in the chart of accounts
        - sort transactions by account, date, and trans_id
        """
        for transaction in transactions:
            if transaction.trans_id.isna().any():
                transaction.trans_id = self._id_counter()

        transactions = pd.concat(transactions)

        categories = list(self.chart.accounts_in_order())
        missing_accounts = pd.Index(transactions.account.unique()).difference(list(self.chart.accounts_in_order()))
        if missing_accounts.size > 0:
            raise ValueError(f'Accounts {missing_accounts} not found in chart of accounts')

        transactions.account = pd.Categorical(transactions.account, categories=categories, ordered=True)
        transactions = transactions.sort_values(['account', 'date', 'trans_id'])
        transactions = transactions.reset_index(drop=True)

        return transactions

    def subledger(self, account:str=None, all_subs:bool=False, **kwargs):
        if account:
            kwargs['account'] = account
        elif not kwargs:
            raise ValueError("No filter criteria provided for subledger")

        missing_keys = [key for key in kwargs if key not in self.transactions.columns]
        if missing_keys:
            raise ValueError(f"Keywords {missing_keys} not found in ledger")

        if all_subs:
            kwargs['account'] = list(nx.descendants(self.chart, kwargs['account'])) + [kwargs['account']]

        conditions = []
        for key, value in kwargs.items():
            if isinstance(value, (list, set, tuple)):
                conditions.append(self.transactions[key].isin(value))
            elif isinstance(value, tuple) and len(value) == 2:
                # Assuming tuple is a range
                conditions.append((self.transactions[key] >= value[0]) & (self.transactions[key] <= value[1]))
            else:
                conditions.append(self.transactions[key] == value)

        subledger = self.transactions[np.logical_and.reduce(conditions)]
        
        return Ledger(self.chart, subledger)
    
    def find_transaction(self, trans_id:str):
        return self.transactions[self.transactions.trans_id == trans_id]

def find_coordinates_multi_array(df, values):
    """
    Find the coordinates of values in a multi-dimensional array
    """ 
    # Convert DataFrame to numpy array for faster operations
    df_array = df.values

    # Create a boolean mask where any of the values match
    value_mask = np.isin(df_array, values)

    # Create a boolean mask where the right adjacent cell is '---' or it's the last column
    right_condition = np.column_stack((df_array[:, 1:] == '---', np.ones(df_array.shape[0], dtype=bool)))

    # Combine the conditions
    combined_mask = value_mask & right_condition

    # Get the coordinates where the combined condition is True
    coordinates = np.argwhere(combined_mask)

    # Get the values at these coordinates
    found_values = df_array[coordinates[:, 0], coordinates[:, 1]]

    value_rows = [coordinates[found_values == value][0,0] for value in values]

    # Check for empty arrays and arrays with more than one item
    # lengths = np.array([len(rows) for rows in value_rows])
    # empty_mask = lengths == 0
    # multiple_mask = lengths > 1

    # Prepare error messages
    # error_messages = []
    # if np.any(empty_mask):
    #     empty_values = np.array(values)[empty_mask]
    #     error_messages.extend(f"Error: Empty list for key '{value}'" for value in empty_values)

    # if np.any(multiple_mask):
    #     multiple_values = np.array(values)[multiple_mask]
    #     multiple_rows = np.array(value_rows)[multiple_mask]
    #     error_messages.extend(f"Error: More than one item for key '{value}': {rows.tolist()}"
    #                           for value, rows in zip(multiple_values, multiple_rows))

    # # If there are any error messages, raise an exception
    # if error_messages:
    #     raise ValueError("\n".join(error_messages))

    return value_rows

def make_multi_level_index_from_chart(chart_index, stat_index:typing.Union[pd.Index, pd.MultiIndex]):
    assert not any(chart_index.value_counts() > 1)
    

    if stat_index.nlevels > 1:
        values = stat_index.get_level_values('account')
    else:
        values = stat_index

    row_idxs = find_coordinates_multi_array(chart_index, values)

    if stat_index.nlevels > 1:
        idx_frame = pd.concat([
            chart_index.loc[row_idxs].reset_index(drop=True),
            stat_index.to_frame().reset_index(drop=True).drop('account', axis=1)
        ], axis=1
        )
    else:
        idx_frame = chart_index.loc[row_idxs]

    return pd.MultiIndex.from_frame(idx_frame)

def shape_metric(name, func, stat):
    metric = func(stat).to_frame().rename(columns={0: name}).T
    if isinstance(stat.index, pd.MultiIndex):
        metric.index = pd.MultiIndex.from_tuples([[name] + ['---'] * (len(stat.index.names) - 1)])
    else:
        metric.index = [name]
    
    return metric

def append_metrics(metric_funcs, stat):
    metrics = [shape_metric(name, func, stat) for name, func in metric_funcs.items()]
    return pd.concat([stat] + metrics)

def loc_by_account(stat, account, drop_levels:bool=True):
    cells = np.argwhere(stat.index.to_frame().reset_index(drop=True).values == account)
    rows = cells[:, 0]
    col_candidates = np.unique(cells[:, 1])
    assert col_candidates.size == 1
    col = col_candidates[0]

    stat = stat.iloc[rows]
    if drop_levels:
        stat = stat.droplevel(stat.index.names[:col+1])

    return stat

class Statement:
    def __init__(self, ledger:Ledger):
        self.ledger = ledger

    def root(self, index:typing.Union[typing.List[str], str]='account', freq:str=None, flat:bool=True, observed:bool=False):
        if isinstance(index, list) and len(index) > 1 and flat:
            warnings.warn('You should NOT provide multiple indexes when flat=True')

        # When observed=True, any NaNs values in the index columns will eliminate the row from the pivot table, so we need to fill NaNs with '---'
        traxns = self.ledger.transactions.copy()
        if observed and isinstance(index, list):
            for idx in index:
                if idx != 'account':
                    traxns.loc[:, idx] = traxns.loc[:, idx].fillna('---')

        stat = pd.pivot_table(traxns, values='amount', index=index, columns='date', aggfunc='sum', observed=observed)
        stat = stat.fillna(0)

        stat.index = make_multi_level_index_from_chart(self.ledger.chart.as_frame(), stat.index)
        stat = stat.sort_index()

        if flat:
            groupbys = []
            for index_col in range(len(stat.index.names)):
                groupby = stat.groupby(level=index_col).sum()
                groupbys.append(groupby)
            stat = pd.concat(groupbys)

        if freq:
            stat = stat.T.resample(freq).sum().T

        return stat
    
    def from_template(self, 
        template:list[str], 
        metrics:dict={}, 
        ratios:dict={}, 
        flip_sign:typing.Union[bool, list[str]]=False, 
        *args, **kwargs
    ):
        stat = self.root(*args, **kwargs)

        if flip_sign:
            if isinstance(flip_sign, bool):
                stat *= -1
            else:
                stat.loc[flip_sign] *= -1

        stat = append_metrics(metrics, stat)
        stat = append_metrics(ratios, stat)

        return stat.loc[template]
