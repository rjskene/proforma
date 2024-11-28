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

def networkx_to_treelib_string(G):
    """
    Convert a NetworkX graph to a string representation using treelib
    """
    def build_tree_string(node, prefix="", is_last=False, is_root=False):
        output = []

        # Add current node
        if is_root:
            output.append(f"{node}")
        else:
            output.append(f"{prefix}{'└── '}{node}")

        # Get children (successors) of the current node
        children = list(G.successors(node))

        # Recursively process children
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            new_prefix = prefix + "    "
            output.extend(build_tree_string(child, new_prefix, is_last_child))

        return output

    # Find all root nodes (nodes with no incoming edges)
    roots = [node for node in G.nodes() if G.in_degree(node) == 0]

    # Build the forest string
    forest = []
    for i, root in enumerate(roots):
        is_last_root = (i == len(roots) - 1)
        forest.extend(build_tree_string(root, is_last=is_last_root, is_root=True))
        if not is_last_root:
            forest.append("")  # Add a blank line between root trees

    # Return the full forest string
    return "\n".join(forest)

BASE_ACCOUNTS = ('Assets', 'Capital')
class Chart(nx.DiGraph):        
    """
    A Chart of Accounts is a directed graph with nodes as accounts and edges as relationships between accounts
    """
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'Chart of Accounts'
        super().__init__(self, *args, **kwargs)
        
        for account_name in self.BASE_ACCOUNTS:
            self.add_node(account_name, attr_name=account_name.lower())

        self.add_subs('Capital', 'Liabilities', 'Equity')
        self.add_subs('Equity', 'Retained Earnings')
        self.add_subs('Retained Earnings', 'Income', 'Expenses')

            
    def __getattr__(self, key):
        nodes = [x for x,y in self.nodes(data=True) if y['attr_name']==key]
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

    def sub_accounts(self, parent):
        return self.successors(parent)
    
    def _make_attr_name(self, name):
        if not isinstance(name, str):
            raise ValueError('Sub-account names should be str')
            
        if ' ' in name:
            name = name.replace(' ', '_')
        return name.lower()
    
    def _tree_repr_(self):
        if len(self.nodes) > 20:
            return 'CHART OF ACCOUNTS'
        
        forest_string = networkx_to_treelib_string(self)

        return 'CHART OF ACCOUNTS\n\n' + forest_string
    
    # CRUD for Graph
    def add_sub(self, parent, sub, **kwargs):
        if parent not in self.find_accounts() and not self.in_degree(parent):
            raise ValueError(f'Parent account "{parent}" must be a base account or have a parent')
        
        if sub in self.nodes():
            raise ValueError(f'Node "{sub}" already exists in the chart')

        if 'attr_name' not in kwargs:
            kwargs['attr_name'] = self._make_attr_name(sub) 

        self.add_node(sub, **kwargs)
        self.add_edge(parent, sub)
        
    def add_subs(self, parent, *subs, **kwargs):
        if parent not in self.find_accounts() and not self.in_degree(parent):
            raise ValueError(f'Parent account "{parent}" must be a base account or have a parent')

        existing_nodes = [sub for sub in subs if sub in self.nodes()]
        if existing_nodes:
            raise ValueError(f'Nodes {existing_nodes} already exist in the chart')

        subs_with_attrs = ((sub, {'attr_name': self._make_attr_name(sub)}) for sub in subs)
        edges = ((parent, sub) for sub in subs)
        
        self.add_nodes_from(subs_with_attrs)
        self.add_edges_from(edges)
        
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
    def from_dict_of_lists(cls, dict_of_lists):
        chart = cls()
        graph = nx.from_dict_of_lists(dict_of_lists, create_using=nx.DiGraph)
        chart.add_nodes_from(graph.nodes(data=True))
        chart.add_edges_from(graph.edges())

        for node in chart.nodes:
            if 'attr_name' not in chart.nodes[node]:
                chart.nodes[node]['attr_name'] = chart._make_attr_name(node)
        
        # if chart.find_accounts() != chart.BASE_ACCOUNTS:
        #     diff_base = set(list(chart.BASE_ACCOUNTS)) - set(chart.find_accounts())
        #     diff_chart = set(chart.find_accounts()) - set(list(chart.BASE_ACCOUNTS))
        #     raise ValueError(f'Chart must contain base accounts {diff_base} and not contain {diff_chart}')
        
        return chart
    
class Entry(dict):
    def __init__(self, account, date, amount=0, trans_id:str=None, **kwargs):
        date = self._format_date(date)

        super().__init__(account=account, date=date, amount=amount, trans_id=trans_id, **kwargs)

        for key, value in kwargs.items():
            self[key] = value

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

    def __repr__(self):
        return self.transactions.__repr__()
    
    def __str__(self):
        return self.transactions.__str__()

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

        return transactions

    def subledger(self, account:str):
        subledger = self.transactions[self.transactions.account == account]
        return Ledger(self.chart, subledger)
    
    def find_transaction(self, trans_id:str):
        return self.transactions[self.transactions.trans_id == trans_id]
