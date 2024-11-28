"""
Set of utility functions for importing QuickBooks data
"""
import datetime as dt
import numpy as np
import pandas as pd
from proforma.proforma import Chart

def append_parent(chart:pd.DataFrame, parent_name:str, subs:list):
    """
    Append a parent account to the chart of accounts
    """
    subs_bool = sum(list(chart.Account.str.contains(f'{acct}:') for acct in subs)).astype(bool)
    chart.loc[subs_bool, 'Account'] = parent_name + ':' + chart.Account.loc[subs_bool]
    
def df_to_dict_of_lists(df):
    """
    Convert a DataFrame to a dictionary of lists
    """
    result = {}
    columns = df.columns.tolist()

    for i in range(len(columns)):
        current_col = columns[i]
        next_col = columns[i+1] if i+1 < len(columns) else None

        for value in df[current_col].unique():
            if value is not None:
                if value not in result:
                    result[value] = []

                if next_col is None:
                    next_values = []
                else:
                    next_values = df[df[current_col] == value][next_col].unique().tolist()
                    next_values = list(filter(lambda a: a is not None, next_values))

                result[value].extend(next_values)
    return result

def read_coa_and_create_dict_of_lists(filename):
    """
    Read a chart of accounts from an Excel file and create a dictionary of lists.

    This function reads a chart of accounts from an Excel file, processes the data to standardize account types,
    appends parent accounts to the chart, and converts the chart to a dictionary of lists format.

    Args:
        filename (str): The name of the Excel file (without the .xlsx extension) containing the chart of accounts.

    Returns:
        dict: A dictionary where each key is an account and the corresponding value is a list of sub-accounts.
    """

    df_chart = pd.read_excel(f'{filename}.xlsx')

    df_chart = df_chart.loc[:, ~df_chart.isna().all()]
    df_chart.Type = df_chart.Type \
        .str.replace('Bank', 'Cash') \
        .str.replace('Cost of Goods Sold', 'COGS') \
        .str.replace('Accounts Receivable', 'AR') \
        .str.replace('Accounts Payable', 'AP') # Needed b/c Type and Account are the same name


    df_chart.loc[df_chart.Type == 'Income', 'Type'] = df_chart.Type.where(df_chart.Type != 'Income', 'Revenue')
    df_chart.loc[df_chart.Type == 'Expense', 'Type'] = df_chart.Type \
        .where(df_chart.Type != 'Expense', 'Operating Expense')

    df_chart.Account = df_chart.Type + ':' + df_chart.Account

    parent_subs = {
        'Current Assets': ['Cash', 'AR', 'Other Current Asset'],
        'Assets':  ['Current Assets', 'Fixed Asset', 'Other Asset'],
        'Current Liabilities': ['AP', 'Credit Card', 'Other Current Liability'],
        'Liabilities': ['Current Liabilities', 'Long Term Liability'],
        'Capital': ['Liabilities', 'Equity'],
        'Income': ['Revenue', 'Other Income'],
        'Expenses': ['Cost of Goods Sold', 'Operating Expense', 'Other Expense']
    }

    for parent, subs in parent_subs.items():
        append_parent(df_chart, parent, subs)

    qb_chart = pd.DataFrame(df_chart.Account.str.split(':').tolist())

    dict_of_lists = df_to_dict_of_lists(qb_chart)

    return dict_of_lists

def read_bsheet(excel_file, sheet_name):
    """
    Read a balance sheet from an Excel file

    This function reads a balance sheet from an Excel file and processes the data to ensure it is in the correct format.
    It handles the following tasks:
    1. Reads the specified sheet from the Excel file.
    2. Removes the last two columns from the data.
    3. Flips the signs of the values in the balance sheet after the 'TOTAL ASSETS' row.
    4. Removes rows that contain totals for specific accounts.
    5. Forward fills missing values in the account columns.
    6. Sets the first row as the header and the 'Account' column as the index.
    7. Drops any rows with missing values.
    8. Combines 'Retained Earnings' and 'Net Income' if both are present.
    9. Ensures the balance sheet sums to zero.

    Args:
        excel_file (str): The path to the Excel file containing the balance sheet.
        sheet_name (str): The name of the sheet within the Excel file to read.

    Returns:
        pd.Series: A pandas Series representing the processed balance sheet, indexed by account.
    """
    bsheet = pd.read_excel(excel_file, sheet_name=sheet_name)

    bsheet = bsheet.iloc[:, :-2]
    flip_signs = bsheet.iloc[np.where(bsheet.iloc[:, 0] == 'TOTAL ASSETS')[0][0] + 1:, -1]
    bsheet.loc[flip_signs.index, bsheet.columns[-1]] *= -1
    # Split the single line of code into discrete steps without using a function
    mask = []

    for index, row in bsheet.iterrows():
        row_contains_total = False
        for cell in row:
            if isinstance(cell, str):
                if cell.startswith('Total ') or cell.startswith('TOTAL '):
                    cell_content = cell.replace('Total ', '').replace('TOTAL ', '')
                    if cell_content in bsheet.iloc[:index, np.where(row == cell)[0][0]].values:
                        row_contains_total = True
                        break
        mask.append(not row_contains_total)

    bsheet = bsheet[mask]
    bsheet.loc[:, bsheet.columns[:-1]] = bsheet.iloc[:, :-1].ffill(axis=1)
    bsheet.iloc[0, -2] = 'Account'
    bsheet.columns = bsheet.iloc[0]
    bsheet = bsheet.iloc[1:, -2:]
    bsheet = bsheet.set_index('Account').dropna()

    if 'Retained Earnings' in bsheet.index and 'Net Income' in bsheet.index:
        bsheet.loc['Retained Earnings'] += bsheet.loc['Net Income']
        bsheet = bsheet.drop('Net Income')

    bsheet = bsheet.squeeze()

    assert bsheet.sum() == 0

    bsheet.name = dt.datetime.strptime(bsheet.name, '%b %d, %y').date().isoformat()
    # Other exists on the balance sheet for some stupid QBooks reason but not in the COA/GL
    bsheet.index = bsheet.index.str.replace(' - Other', '')

    return bsheet

def read_gl(excel_file, chart:Chart, sheet_name:str='Sheet1'):
    """
    Read a general ledger from an Excel file

    This function reads a general ledger from an Excel file and processes it according to the provided chart of accounts.

    The function performs the following steps:
    1. Reads the Excel file into a pandas DataFrame.
    2. Removes columns with 'Unnamed:' in their header.
    3. Renames specific columns to standardize the DataFrame.
    4. Filters out rows where the 'Account' column is NaN.
    5. Adjusts the 'amount' column to account for credits.
    6. Selects and reorders specific columns.
    7. Asserts that the sum of the 'amount' column is zero.
    8. Identifies accounts in the chart that are not present in the general ledger and adds them with an amount of zero.
    9. Fills missing 'trans_id' values with 'null_entry'.
    10. Fills missing 'date' values with the earliest date in the DataFrame.

    Args:
        excel_file (str): The path to the Excel file containing the general ledger.
        chart (Chart): The chart of accounts to be used for processing.
        sheet_name (str): The name of the sheet within the Excel file to read. Defaults to 'Sheet1'.

    Returns:
        pd.DataFrame: A pandas DataFrame representing the processed general ledger.
    """
    gl = pd.read_excel(excel_file, sheet_name=sheet_name)

    gl = gl.loc[:, ~gl.columns.str.contains('Unnamed:')]
    gl = gl.rename(columns={'Trans #': 'trans_id', 'Date': 'date', 'Account': 'account', 'Debit': 'amount', 'Class': 'class'})
    gl = gl.loc[gl.account.notna()].copy()
    gl.amount = gl.amount.where(gl.Credit.fillna(0) == 0, -gl.Credit)
    gl.amount = gl.amount.fillna(0)
    gl = gl[['trans_id', 'date', 'account', 'amount', 'class']].copy()

    assert abs(gl.amount.sum()) < 1e-6

    return gl
