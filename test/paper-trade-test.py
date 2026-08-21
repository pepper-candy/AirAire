from futu import *

# 创建交易上下文（需要指定市场，这里以港股为例）
trd_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.HK, host='127.0.0.1', port=11111)

# 1. 查询模拟账户信息（正确方法：accinfo_query）
ret, data = trd_ctx.accinfo_query(trd_env=TrdEnv.SIMULATE)
if ret == RET_OK:
    print("✅ 模拟账户信息：")
    # 显示总资产、现金、市值等关键字段
    print(data[['total_assets', 'cash', 'market_val', 'currency']])
else:
    print("❌ 获取账户信息失败：", data)

# 2. 模拟下单（这部分之前是对的，保留）
ret, data = trd_ctx.place_order(
    price=445.0,
    qty=100,
    code="HK.00700",
    trd_side=TrdSide.BUY,
    trd_env=TrdEnv.SIMULATE
)

if ret == RET_OK:
    print(f"✅ 模拟下单成功！订单编号：{data['order_id']}")
else:
    print("❌ 模拟下单失败：", data)

trd_ctx.close()