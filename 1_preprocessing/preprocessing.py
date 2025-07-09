"""
This module contains functions to preprocess event logs

Functions:

    sort_log
    debiasing
    mapping_case_id
    add_soc_eoc
    create_time_features
    train_log_normalize
    test_log_normalize
    train_mapping_event_name
    test_mapping_event_name
    
"""

import pandas as pd
import numpy as np

def sort_log(df, 
             timestamp):
    """    
    Transform the format of timestamp and sort the event log by timestamp

    Parameters
    ----------
    df: pandas DataFrame
        Event log
    timestamp: str
        Name of column containing timestamp
                
    Returns
    -------
    sorted_df: pandas DataFrame
        Event log sorted by timestamp

    """
    # convert timestamp to Pandas datetime
    df[timestamp] = pd.to_datetime(df[timestamp], format='mixed').dt.tz_convert('UTC')

    # sort the event log by timestamp
    sorted_df = df.sort_values(by=timestamp).reset_index(drop=True)

    return sorted_df

def debiasing(df,
            start_date,
            end_date,
            max_duration,
            max_len,
            case_id, 
            timestamp):
    """
    Remove chronological outliers and potentially incomplete cases.

    The codes and the setting of parameters (start_date, end_date, max_duration)
    are adapted from the follwing paper:
    Weytjens, H., De Weerdt, J. (2022). Creating Unbiased Public Benchmark 
    Datasets with Data Leakage Prevention for Predictive Process Monitoring. 
    In: Marrella, A., Weber, B. (eds) Business Process Management Workshops. 
    BPM 2021. Lecture Notes in Business Information Processing, vol 436. 
    Springer, Cham.

    Parameters
    ----------
    df: pandas DataFrame
        Event log
    start_date: str 'YYYY-MM'
        Cases starting before this month are removed
    end_date: str 'YYYY-MM'
        Cases ending after this month are removed
    max_duration: float
        Maximum days a normal case lasts
    max_len: int
        Maximum number of events a normal case contains (98.5th percentile).
    case_id: str
        Name of column containing case ID
    timestamp: datetime
        Name of column containing timestamp

    Returns
    -------
    df: pandas Dataframe
        Debiased event log  
    latest_start: datetime
        The last timestamp minus the max_duration, the ending point of test set

    """
    # step 1: remove outliers starting before start_date
    if start_date is not None:
        # create tables containing the first timestamp of each case
        case_starts_df = df.groupby(case_id)[timestamp].min().reset_index()
        # convert timestamp to 'YYYY-MM'
        case_starts_df['date'] = case_starts_df[timestamp].dt.to_period('M')
        # create array containing case_id of cases starting after start_date 
        cases_after = case_starts_df[case_starts_df['date'].astype('str') >= start_date][case_id].values
        # filter the dataframe
        df = df[df[case_id].isin(cases_after)].reset_index(drop=True)

    # step 2: remove outliers ending after end_date
    if end_date is not None: 
        # create tables containing the last timestamp of each case
        case_stops_df = df.groupby(case_id)[timestamp].max().reset_index()
        # convert timestamp to 'YYYY-MM'
        case_stops_df['date'] = case_stops_df[timestamp].dt.to_period('M')
        # create array containing case_id of cases ending before end_date
        cases_before = case_stops_df[case_stops_df['date'].astype('str') <= end_date][case_id].values
        # filter the dataframe
        df = df[df[case_id].isin(cases_before)].reset_index(drop=True)

    # step 3: retain only cases shorter than max_duration
    # compute each case's duration in days
    agg_dict = {timestamp: ['min', 'max']}
    duration_df = df.groupby(case_id).agg(agg_dict).reset_index()
    duration_df["duration"] = (duration_df[(timestamp,"max")] - \
                               duration_df[(timestamp,"min")]).dt.total_seconds() \
                                / (24 * 60 * 60)
    # create array containing case_id of cases whose duration is less than max_duration
    cases_retained = duration_df[duration_df["duration"] <= max_duration * 1.00000000001][case_id].values
    # filter the dataframe
    df = df[df[case_id].isin(cases_retained)].reset_index(drop=True)

    # step 4: drop cases starting after the latest_start (potentially incomplete cases)
    # latest_start: the last timestamp minus the max_duration
    latest_start = df[timestamp].max() - pd.Timedelta(max_duration, unit='D')
    # create array containing case_id of cases starting before latest_start
    cases_retained = duration_df[duration_df[(timestamp, "min")] <= latest_start][case_id].values
    # filter the dataframe
    df = df[df[case_id].isin(cases_retained)].reset_index(drop=True)

    # step 5: drop cases longer than max_len
    # calculate the length of each case
    event_counts = df.groupby(case_id).size().reset_index(name='case_length')
    # create array containing case_id of cases shorter than max_len
    cases_retained = event_counts[event_counts['case_length'] <= max_len][case_id].values
    # filter the dataframe
    df = df[df[case_id].isin(cases_retained)].reset_index(drop=True)

    # ensure event log is sorted by timestamp
    df = df.sort_values(by=timestamp).reset_index(drop=True)
    
    return df, latest_start 

def mapping_case_id(df, 
                    case_id):
    """
    
    Create a dictionary that stores the one-to-one mapping between case_id and 
    index, then transform the case_id into index.  
    
    Parameters
    ----------
    df: pandas DataFrame
        Event log 
    case_id: str
        Name of column containing case ID

    Returns
    -------
    df: pandas DataFrame
        Dataframe with case ID transformed into index.
    case_id_dict: dictionary
        Store the one-to-one mapping between case_id and index
    
    """
    # create the mapping dictionary
    case_id_dict = {}
    n = 1
    for id in df[case_id].unique():
        case_id_dict[id] = n
        n += 1
    
    # map case ID to index
    df[case_id] = df[case_id].map(case_id_dict)

    return df, case_id_dict

def add_soc_eoc(df,
                case_id, 
                timestamp, 
                event_name):
    
    """
    
    Create rows containing SOC (Start of Case) token and EOC (End of Case) token. 
    
    Parameters
    ----------
    df: pandas.DataFrame
        Event log 
    case_id: str
        Name of column containing case ID
    timestamp: str
        Name of column containing timestamp
    event_name: str
        Name of column containing activity label

    Returns
    -------
    df: pandas.DataFrame
        Two rows (SOC, EOC) are added for each case, and one column (event_idx) 
        is added
    
    """
    # sort dataframe by case_id and timestamp
    df = df.sort_values(by=[case_id, timestamp]).reset_index(drop=True)

    # add SOC and EOC rows
    new_rows = []
    for i in range(len(df)):

        # if this is the first event in a case, append a soc_row before 
        # appending all events of the case
        if i == 0 or df[case_id].iloc[i] != df[case_id].iloc[i - 1]:

            # Create a new row where event name is 'SOC', and values in other 
            # columns are the same as the first event
            soc_row = df.iloc[i].copy() # This returns a series
            soc_row[event_name] = 'SOC'
            
            # Append the 'SOC' row 
            new_rows.append(soc_row)

        # append other rows in this case
        new_rows.append(df.iloc[i])

        # if this is the last event in a case, append a eoc_row after all events of the case
        if i == len(df) - 1 or df[case_id].iloc[i] != df[case_id].iloc[i + 1]:
            # create a new row where event name is 'EOC', and values in other columns are the same as the last event
            eoc_row = df.iloc[i].copy() # This returns a series
            eoc_row[event_name] = 'EOC'

            # Append the 'EOC' row 
            new_rows.append(eoc_row)

    new_df = pd.DataFrame(new_rows)

    # Create a helper column 'Order' to ensure that SOC is always immediately 
    # above the first event in a case, EOC is always immediately below the last 
    # event in a case
    new_df['Order'] = new_df.apply(lambda row: 
                                   row[case_id] * 10 + (1 if row[event_name] == 'SOC' 
                                                    else (3 if row[event_name] == 'EOC' 
                                                    else 2)), 
                                    axis=1)
    new_df = new_df.sort_values(by=[timestamp, 'Order']).reset_index(drop=True)
    # Drop the helper column
    new_df = new_df.drop(columns=['Order'])
    # NOT to sort event log again by timestamp, otherwise the positions of SOC
    # and EOC cannot be retained.
    # After this step, never sort the event log by timestamp until creating 
    # prefix/suffix tensor. If sort is needed, sort by event_idx

    # Create event index based on the chronological order 
    new_df['event_idx'] = range(1, len(new_df) + 1)

    return new_df

def create_time_features(df, 
                        case_id, 
                        timestamp,
                        event_idx):
    """
    
    Create three temporal fetures: 
    - log_ts_pre: time since the previou event in the event log
    - trace_ts_pre: time since the previous event in the case
    - trace_ts_statr: time since the start (i.e. the first event) of the case
    
    Parameters
    ----------
    df: pandas DataFrame
        Event log 
    case_id: str
        Name of column containing case ID
    timestamp: datetime
        Name of column containing timestamp
    event_idx: str
        Index of events in event log

    Returns
    -------
    df: pandas DataFrame
        Event log with three temporal features
    
    """
    # ensure the event log is sorted by event_idx
    df = df.sort_values(by=event_idx).reset_index(drop=True)

    # calulate time since the previous event in event log
    df['log_ts_pre'] = df[timestamp].diff(periods=1).dt.total_seconds()

    # for the first event in the event log, log_ts_pre is set as 0
    df.loc[0, 'log_ts_pre'] = 0.0

    # calculate time since the previous event in case
    df = df.sort_values(by=[case_id, timestamp]).reset_index(drop=True)
    df['trace_ts_pre'] = 0.0
    for i in range(1, len(df)): # start from 1 to avoid calculating i-1 when i=0
        if df[case_id].iloc[i] == df[case_id].iloc[i - 1]: # if it is not the first event in each case 
            df.loc[i, 'trace_ts_pre'] = (df[timestamp].iloc[i] - df[timestamp].iloc[i - 1]).total_seconds()

    # a helper column containing the start time of each case
    case_start_times_df = df.groupby(case_id)[timestamp].min().\
        to_frame(name='case_start_time').reset_index(names=case_id) 
    df = pd.merge(df, case_start_times_df, on=case_id, how='inner')

    # calculte time since the start of the case
    df['trace_ts_start'] = (df[timestamp] - df['case_start_time']).dt.total_seconds()

    # drop the helper column
    df = df.drop(columns=['case_start_time'])
    df = df.sort_values(by=event_idx).reset_index(drop=True)

    return df


def train_log_normalize(df, 
                     continuous_features):
    """
    Apply log transformation and max-min normalization to continuous features in 
    the training set.

    Parameters
    ----------
    df: pandas Dataframe
        Event log
    continuous_features: list
        Name(s) of column(s) containing continuous features
    
    Returns
    -------
    df: pandas Dataframe
        Event log with log normalized continuous features
    min_dict:dictionary
        A dictionary where keys are feature names and values are their minimum 
        values
    max_dict: dictionary
        A dictionary where keys are feature names and values are their maximum 
        values
    """
    # initialize dictionaries to store min and max for each feature
    max_dict = {}
    min_dict = {}

    # loop through all features
    for col in continuous_features:
        # transform x into ln(1+x), since x could be 0
        df[col] = np.log1p(df[col])

        col_max = df[col].max()
        col_min = df[col].min()
        
        # store the max and min in the dictionaries
        max_dict[col] = col_max
        min_dict[col] = col_min
        
        # log-normalize the column
        df[col] = (df[col] - col_min) / (col_max - col_min)
    
    return df, max_dict, min_dict

def test_log_normalize(df, 
                       max_dict, 
                       min_dict,
                     continuous_features):
    """
    Apply log transformation and max-min normalization to continuous features in 
    the test set, using max and min calculated from the training set.

    Parameters
    ----------
    df: pandas Dataframe
        Event log
    max_dict: dictionary
        A dictionary where keys are feature names and values are their minimum 
        values
    min_dict: dictionary
        A dictionary where keys are feature names and values are their maximum 
        values
    continuous_features: list
        Name(s) of column(s) containing continuous features.

    Returns
    -------
    df: pandas Dataframe
        Event log with log normalized continuous features
    """
    # loop through all features
    for col in continuous_features:
        col_max = max_dict[col]
        col_min = min_dict[col]
        
        # log-normalize the column
        df[col] = np.log1p(df[col])
        df[col] = (df[col] - col_min) / (col_max - col_min)
    
    return df

def train_mapping_event_name(df, 
                             event_name):
    """
    
    Mapping activity labels in the training set to index.
    
    Parameters
    ----------
    df: pandas DataFrame
        Training set
    event_name: str
        Name of column containing activity label

    Returns
    -------
    df: pandas DataFrame
        Event log
    event_name_dict: dictionary
        A dictionary where keys are activity labels and values are their indices.
    
    """
    # initialize the dictionary
    event_name_dict = {"SOC":2, 
                       "EOC":3}
    
    # create the mapping dictionary
    n = int(4)
    for name in df[event_name].unique():
        if name not in event_name_dict.keys():
            event_name_dict[name] = n
            n += 1
    
    # map activity label to index
    df[event_name] = df[event_name].map(event_name_dict)

    return df, event_name_dict


def test_mapping_event_name(df, 
                            event_name_dict,
                            event_name):
    """
    
    Mapping activity labels in test set to index.
    
    Parameters
    ----------
    df: pandas DataFrame
        Event log 
    event_name_dict: dictionary
        A dictionary where keys are activity labels and values are their indices.
    event_name: str
        Name of column containing activity label

    Returns
    -------
    df: pandas DataFrame
        Event log
    event_name_dict: dictionary
        The keys are activity labels in training set and the values are the corresponding index.
    
    """
    # initialize the dictionary
    test_event_name_dict = event_name_dict.copy()
    
    # create the mapping dictionary
    for name in df[event_name].unique():
        # activity labels not appearing in training set will be assigned index 1
        if name not in test_event_name_dict.keys():
            test_event_name_dict[name] = int(1)
    
    # map activity label to index
    df[event_name] = df[event_name].map(test_event_name_dict)

    return df, test_event_name_dict