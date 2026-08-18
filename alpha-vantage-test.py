"""
Alpha Vantage API 连接测试脚本
测试学术权限是否生效，新闻情绪数据是否能正常拉取
"""

import os
import requests
from dotenv import load_dotenv

# 尝试加载 .env 文件（如果存在）
load_dotenv()

# 配置
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()

# 测试用的股票代码（Alpha Vantage 格式）
TEST_TICKERS = {
    "HK.00700": "TCEHY",    # 腾讯 ADR
    "US.COST": "COST",      # Costco
    "US.KO": "KO",          # Coca-Cola
}

def test_alpha_vantage(api_key: str) -> dict:
    """测试 Alpha Vantage API 连接和新闻情绪功能"""
    
    results = {}
    
    if not api_key:
        print("❌ 错误: 未找到 API Key。")
        print("   请通过以下方式之一提供:")
        print("   1. 在项目根目录创建 .env 文件，添加 ALPHAVANTAGE_API_KEY=你的密钥")
        print("   2. 直接在此脚本中设置 API_KEY 变量")
        return results
    
    print(f"🔑 使用 API Key: {api_key[:4]}...{api_key[-4:]}")
    print("=" * 60)
    
    # 测试单个股票
    test_ticker = "COST"
    print(f"📡 测试股票: {test_ticker}")
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": test_ticker,
        "limit": 5,  # 只获取5条新闻
        "apikey": api_key
    }
    
    try:
        print(f"🔄 正在请求: {url}")
        print(f"📦 参数: {params}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # 检查是否有错误信息
        if "Error Message" in data:
            print(f"❌ API 返回错误: {data['Error Message']}")
            results[test_ticker] = {"status": "error", "message": data["Error Message"]}
            return results
        
        # 检查是否到达请求限制
        if "Note" in data and "API call frequency" in data["Note"]:
            print(f"⚠️  API 频率限制: {data['Note']}")
            results[test_ticker] = {"status": "rate_limited", "message": data["Note"]}
            return results
        
        # 解析新闻数据
        feed = data.get("feed", [])
        sentiment_scores = []
        
        print(f"\n📰 获取到 {len(feed)} 条新闻:")
        for i, item in enumerate(feed[:3], 1):  # 只显示前3条
            title = item.get("title", "无标题")[:60]
            overall_sentiment = item.get("overall_sentiment_score", 0)
            print(f"  {i}. {title}...")
            print(f"     情绪分数: {overall_sentiment:.2f}")
            sentiment_scores.append(overall_sentiment)
        
        # 计算平均情绪
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        print(f"\n📊 平均情绪分数: {avg_sentiment:.3f} (范围: -1 到 1)")
        
        # 检查是否有 ticker_sentiment
        if feed:
            first_item = feed[0]
            ticker_sent = first_item.get("ticker_sentiment", [])
            if ticker_sent:
                for ts in ticker_sent:
                    if ts.get("ticker") == test_ticker:
                        print(f"🎯 {test_ticker} 专属情绪分数: {ts.get('ticker_sentiment_score', 'N/A')}")
        
        results[test_ticker] = {
            "status": "success",
            "news_count": len(feed),
            "avg_sentiment": avg_sentiment,
            "data": data
        }
        
        print("\n✅ Alpha Vantage API 测试成功！")
        
    except requests.exceptions.Timeout:
        print("❌ 请求超时，请检查网络连接")
        results[test_ticker] = {"status": "timeout"}
    except requests.exceptions.ConnectionError:
        print("❌ 网络连接失败，请检查网络")
        results[test_ticker] = {"status": "connection_error"}
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP 错误: {e}")
        results[test_ticker] = {"status": "http_error", "message": str(e)}
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        results[test_ticker] = {"status": "unknown_error", "message": str(e)}
    
    return results


def test_batch_tickers(api_key: str):
    """批量测试多只股票"""
    print("\n" + "=" * 60)
    print("📊 批量测试多只股票")
    print("=" * 60)
    
    for ticker, av_symbol in TEST_TICKERS.items():
        print(f"\n🔍 测试 {ticker} ({av_symbol})")
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": av_symbol,
            "limit": 1,
            "apikey": api_key
        }
        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            feed = data.get("feed", [])
            sentiment = 0
            if feed:
                sentiment = feed[0].get("overall_sentiment_score", 0)
            print(f"   ✅ 获取到 {len(feed)} 条新闻，情绪: {sentiment:.2f}")
        except Exception as e:
            print(f"   ❌ 失败: {e}")


if __name__ == "__main__":
    print("🚀 Alpha Vantage API 测试工具")
    print("=" * 60)
    
    # 如果没有在 .env 中找到，直接在这里输入
    if not API_KEY:
        print("\n📝 未在 .env 中找到 API Key")
        manual_key = input("请粘贴你的 Alpha Vantage API Key (直接按 Enter 跳过): ").strip()
        if manual_key:
            API_KEY = manual_key
    
    if not API_KEY:
        print("\n❌ 没有 API Key，无法测试")
        print("   请将 API Key 添加到 .env 文件或直接输入")
        exit(1)
    
    # 运行测试
    results = test_alpha_vantage(API_KEY)
    
    # 如果单股票测试成功，批量测试
    if results and results.get("COST", {}).get("status") == "success":
        test_batch_tickers(API_KEY)
    
    print("\n" + "=" * 60)
    print("测试完成！")