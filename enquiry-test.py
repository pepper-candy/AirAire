from futu import *

quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
ret, data = quote_ctx.get_market_snapshot(['HK.00700'])
if ret == RET_OK:
    print("✅ 行情获取成功！腾讯当前价格：", data['last_price'].values[0])
    print("🎉 恭喜，你的免费 LV1 权限已经生效了！")
else:
    print("❌ 获取失败，错误信息：", data)
    print("🔍 如果提示 '无权限'，说明还需要在商城里找免费的 LV1 订阅一下。")
quote_ctx.close()