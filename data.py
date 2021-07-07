from web3 import Web3
import json

import requests
from pprint import pprint

from datetime import datetime
import pause

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
