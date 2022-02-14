import pandas as pd
import json
import zipfile
from glob import glob
from tqdm import tqdm
import numpy as np
from datetime import timedelta

def map_caseRate_ZCTA_to_CT():
    """
    DEPRECATED
    :return: no return, write a csv under data
    """
    caseRate = pd.read_csv('../data/caserate-by-modzcta.csv')

    # gather column, rename, parse zip code
    caseRate = pd.melt(caseRate, id_vars = 'week_ending')
    caseRate['variable'] = caseRate['variable'].str.split('_', expand = True).loc[:,1]
    caseRate = caseRate.rename({'variable': 'ZIP', 'value': 'caseRate'}, axis=1)
    caseRate.ZIP = caseRate.ZIP.astype('str')

    # exclude aggregated statistics row
    caseRate = caseRate.loc[~caseRate.ZIP.isin(['SI','QN','MN','BK','BX','CITY']),:]

    crosswalk = pd.read_excel('../data/ZIP_TRACT_crosswalk.xlsx')
    crosswalk['ZIP'] = crosswalk['ZIP'].astype('str').str.pad(5, side='left', fillchar='0')

    # join tables
    caseRate_cw = caseRate.merge(crosswalk, how = 'left', on = 'ZIP')
    caseRate_cw['rateShare'] = caseRate_cw['caseRate'] * caseRate_cw['TOT_RATIO']

    # aggregate to census tract
    caseRateByCT = caseRate_cw.groupby(['week_ending','TRACT']).agg(rate = ('rateShare','sum'))
    caseRateByCT = caseRateByCT.reset_index()

    caseRateByCT.to_csv('../data/caserate_by_tract.csv', index = False)


def write_caseRate_by_ZCTA():
    """

    :return: no return, write a csv under data
    """
    caseRate = pd.read_csv('../data/caserate-by-modzcta.csv')

    # gather column, rename, parse zip code
    caseRate = pd.melt(caseRate, id_vars = 'week_ending')
    caseRate['variable'] = caseRate['variable'].str.split('_', expand = True).loc[:,1]
    caseRate = caseRate.rename({'variable': 'ZIP', 'value': 'caseRate'}, axis=1)
    caseRate.ZIP = caseRate.ZIP.astype('str')

    # exclude aggregated statistics row
    caseRate = caseRate.loc[~caseRate.ZIP.isin(['SI','QN','MN','BK','BX','CITY']),:]
    caseRate['week_ending'] = pd.to_datetime(caseRate['week_ending']) + timedelta(days = 2)
    caseRate['week_ending'] =  caseRate['week_ending'].astype('str')

    caseRate.to_csv('../data/caserate_by_zcta_cleaned.csv', index=False)


def get_NYC_ZIP_List():
    """

    :return: a numpy array containing all ZIP CODES in NYC
    """
    return pd.read_csv('../data/caserate_by_zcta_cleaned.csv')['ZIP'].astype('str').unique()


def write_NYC_crosswalk():

    crosswalk = pd.read_excel('../data/TRACT_ZIP_crosswalk.xlsx')
    crosswalk.ZIP = crosswalk.ZIP.astype('str')
    crosswalk.TRACT = crosswalk.TRACT.astype('str')
    crosswalk = crosswalk.loc[crosswalk.ZIP.isin(get_NYC_ZIP_List()), :]

    crosswalk.to_csv('../data/NYC_TRACT_to_ZIP.csv', index=False)



def unzip_patterns():
    """

    :return: no return, unzip patterns zip to data. (do not push to github folder)
    """
    for file in glob("..\\data\\NY-PATTERNS*.zip"):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall('..\\data\\patterns\\{0}'.format(file.split('\\')[2].split('-')[2]))




def clean_patterns_unaggregated(starting_from = '2020-08-17'):
    """

    :return: no return, write a csv file 'pattern_cleaned.csv' for each month
    """
    cw = pd.read_csv('../data/NYC_TRACT_to_ZIP.csv')
    cw.ZIP, cw.TRACT = cw.ZIP.astype('str'), cw.TRACT.astype('str')
    column_name = ['date_range_end', 'placekey', 'sg_wp2__visitor_home_aggregation']

    df = pd.read_csv('..\\data\\patterns\\NY.csv')
    df = df.loc[df['sg_wp2__visitor_home_aggregation'] != '{}', column_name]
    df = df.loc[df['date_range_end'] >= starting_from, :]
    df = df.rename(columns={'sg_wp2__visitor_home_aggregation': 'visitor'})


    for week in df['date_range_end'].unique():
        print('Processing Week {0}'.format(week))

        df_week = df.loc[df['date_range_end'] == week, :].reset_index(drop = True)
        visitor_list = [json.loads(dict) for dict in df_week.visitor]

        # loop through rows to map ZIP to TRACT
        for i in tqdm(range(df_week.shape[0])):
            if sum([visitor_list[i][x] for x in visitor_list[i]]) < 9 or len(visitor_list[i]) < 3:
                visitor_list[i] = ''
            else:
                dic = pd.DataFrame.from_dict(visitor_list[i], orient = 'index')\
                    .reset_index()\
                    .rename(columns = {'index':'TRACT', 0:'n'})\
                    .merge(cw, how='left', on='TRACT').loc[:, ['ZIP', 'n']]\
                    .dropna()\
                    .groupby('ZIP').agg({'n':'sum'}).reset_index()

                if dic.shape[0] > 2:
                    visitor_list[i] = dict( (dic['ZIP'][i], dic['n'][i]) for i in range(dic.shape[0]))
                else:
                    visitor_list[i] = ''

        df_week['visitor'] = visitor_list
        df_week = df_week.loc[df_week['visitor'] != '',:]

        df_week.to_csv('..\\data\\patterns_cleaned\\patterns_combined_{0}.csv'.format(week), index=False)



