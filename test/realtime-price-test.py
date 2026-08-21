from futu import OpenQuoteContext
q = OpenQuoteContext(host='127.0.0.1', port=11111)
ret, data = q.get_market_snapshot(['HK.00700', 'US.COST'])
print(data)
q.close()