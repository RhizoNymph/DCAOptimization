from numpy import csingle
from web3 import Web3
import json

import requests
from pprint import pprint

from datetime import datetime
import pause

import os
import sqlite3

import pandas as pd

from configparser import ConfigParser

def run_query(subgraph, query):

    request = requests.post(subgraph,
                            '',
                            json={'query': query})
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception('Query failed. return code is {}.      {}'.format(request.status_code, query))

def getPrice(mode='tick'):
  if mode in ['tick', 'date']:
    subgraph_v3 = 'https://api.thegraph.com/subgraphs/name/drcyph3r/uniswap-v3'
    if mode == 'tick':
      query = """
      {
        pools(first:1, where: {id: "0xc2e9f25be6257c210d7adf0d4cd6e3e881ba25f8"}, orderBy: tick, orderDirection: desc) {
          token0Price
        }
      }
      """
    else:
      query = """
      {
        poolDayDatas(first:1, where: {pool: "0xc2e9f25be6257c210d7adf0d4cd6e3e881ba25f8"}, orderBy: date, orderDirection: desc) {
          token0Price
        }
      }
      """

    price = round(float(run_query(subgraph_v3, query)['data']['pools'][0]['token0Price']), 2)
    return price
  else:
    raise Exception('mode must be \'tick\' or \'date\'')

def startup():
  pass
def getMostRecent():
  if os.path.exists('data.db'):
    con = sqlite3.connect('data.db')
    cur = con.cursor()

    # for row in cur.execute('SELECT * FROM closes ORDER BY time'):

def csvToSqlite(db, file, cur, cols, timecol=None, mode='replace'):
  tablename = file.split('.csv')[0]
  tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
  if tablename in tables:
    if mode == 'replace':     
        cur.execute('DROP TABLE :table', {'table': tablename})  
    elif mode == 'fail':
      raise Exception('table exists and mode set to fail')
    elif mode == 'continue':
      return
    else:
      raise Exception('mode must be replace or fail')
  
  df = pd.read_csv(file)
  df = df[cols]

  if timecol != None:
    df[timecol] = pd.to_datetime(df[timecol])
    df.sort_values(by=[timecol])

  df.to_sql(tablename, con)

# MAIN    
if not os.path.exists('data.db'):
  cfg = ConfigParser()
  cfg.read('csvs.cfg')

  dailies = cfg['CSVS']['csvs_1D'].split(',')
  sixhour = cfg['CSVS']['csvs_360'].split(',')

  con = sqlite3.connect('data.db')
  cur = con.cursor()

  for each in dailies+sixhour:
    csvToSqlite('data.db', cur, each, ['time', 'close'], 'time')