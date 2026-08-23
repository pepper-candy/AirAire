import type { Fill, Headline, Snapshot, SnapshotResponse } from "@/lib/types";

function ago(minutes: number): string {
  return new Date(Date.now() - minutes * 60_000).toISOString();
}

function avTime(minutesAgo: number): string {
  const d = new Date(Date.now() - minutesAgo * 60_000);
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  const h = String(d.getUTCHours()).padStart(2, "0");
  const min = String(d.getUTCMinutes()).padStart(2, "0");
  return `${y}${m}${day}T${h}${min}00`;
}

function headlines(
  rows: { title: string; source: string; minutesAgo: number; score: number; slug: string }[],
): Headline[] {
  return rows.map((row) => ({
    title: row.title,
    source: row.source,
    url: `https://example.invalid/news/${row.slug}`,
    time_published: avTime(row.minutesAgo),
    sentiment_score: row.score,
  }));
}

const TENCENT = headlines([
  { title: "Tencent cloud unit posts stronger-than-expected quarter", source: "Reuters", minutesAgo: 18, score: 0.38, slug: "t1" },
  { title: "WeChat mini-program ads stabilize after a quiet July", source: "SCMP", minutesAgo: 42, score: 0.12, slug: "t2" },
  { title: "Gaming pipeline seen as mixed into the holiday window", source: "Bloomberg", minutesAgo: 95, score: -0.08, slug: "t3" },
  { title: "Analysts lift HK.00700 target on buyback pace", source: "Goldman", minutesAgo: 140, score: 0.21, slug: "t4" },
  { title: "China internet names fade into the US cash open", source: "FT", minutesAgo: 200, score: -0.16, slug: "t5" },
  { title: "Tencent Music royalty talks remain a sideshow", source: "Nikkei", minutesAgo: 260, score: 0.04, slug: "t6" },
  { title: "Honor of Kings esports deal extended two years", source: "Variety", minutesAgo: 320, score: 0.19, slug: "t7" },
  { title: "Beijing commentary stays muted on platform names", source: "Caixin", minutesAgo: 400, score: 0.02, slug: "t8" },
  { title: "Tencent holds Meituan stake unchanged this quarter", source: "WSJ", minutesAgo: 480, score: 0.06, slug: "t9" },
  { title: "Cloud capex guidance leaves some room for caution", source: "Barclays", minutesAgo: 560, score: -0.11, slug: "t10" },
  { title: "Share buyback window reopens after the close", source: "HKEX", minutesAgo: 640, score: 0.27, slug: "t11" },
  { title: "Older note: video accounts still growing off a small base", source: "36Kr", minutesAgo: 900, score: 0.09, slug: "t12" },
]);

const MEITUAN = headlines([
  { title: "Meituan delivery margins compress on summer subsidies", source: "Reuters", minutesAgo: 33, score: -0.31, slug: "m1" },
  { title: "Hotel booking rebound helps the in-store line", source: "SCMP", minutesAgo: 88, score: 0.14, slug: "m2" },
  { title: "Keeta overseas losses still the main debate", source: "Bloomberg", minutesAgo: 170, score: -0.22, slug: "m3" },
  { title: "Food delivery volume holds on weekdays", source: "Citi", minutesAgo: 300, score: 0.05, slug: "m4" },
]);

const CATL = headlines([
  { title: "CATL export licenses discussed ahead of EU battery rules", source: "Reuters", minutesAgo: 27, score: -0.18, slug: "c1" },
  { title: "Energy-storage orders keep the book thicker than expected", source: "CNBC", minutesAgo: 110, score: 0.29, slug: "c2" },
  { title: "Lithium carbonate slide is a tailwind for cell makers", source: "S&P", minutesAgo: 210, score: 0.16, slug: "c3" },
  { title: "Tesla contract chatter remains unconfirmed", source: "Electrek", minutesAgo: 390, score: 0.03, slug: "c4" },
  { title: "Hong Kong listing remains the liquidity venue", source: "HKEX", minutesAgo: 510, score: 0.01, slug: "c5" },
  { title: "Idle plant rumors are denied by the company", source: "Yicai", minutesAgo: 700, score: -0.07, slug: "c6" },
]);

const COSTCO = headlines([
  { title: "Costco membership renewals stay unusually sticky", source: "WSJ", minutesAgo: 22, score: 0.41, slug: "k1" },
  { title: "Kirkland mix shifts toward higher-ticket goods", source: "CNBC", minutesAgo: 76, score: 0.18, slug: "k2" },
  { title: "Gasoline traffic is a wash for the quarter", source: "Bloomberg", minutesAgo: 155, score: -0.04, slug: "k3" },
  { title: "US defensive names bid as rates ease a touch", source: "Reuters", minutesAgo: 240, score: 0.11, slug: "k4" },
  { title: "Warehouse openings in Texas stay on the printed calendar", source: "Retail Dive", minutesAgo: 430, score: 0.08, slug: "k5" },
  { title: "A longer note about gold-bar demand that wrapped the weekend tape and still sits in the cache", source: "Barron's", minutesAgo: 620, score: 0.23, slug: "k6" },
  { title: "Private-label snacks quietly gain share", source: "WSJ", minutesAgo: 800, score: 0.07, slug: "k7" },
]);

const KO = headlines([
  { title: "Coca-Cola pricing power holds in Latin America", source: "Reuters", minutesAgo: 40, score: 0.17, slug: "ko1" },
  { title: "Volume in China is described as only okay", source: "Bloomberg", minutesAgo: 130, score: -0.09, slug: "ko2" },
  { title: "Bottler margins are the quiet support", source: "FT", minutesAgo: 280, score: 0.06, slug: "ko3" },
]);

const FILLS: Fill[] = [
  {
    time: ago(12),
    ticker: "HK.00700",
    side: "BUY",
    qty: 200,
    price: 412.2,
    reason: "Action: Buy 200 Tencent. Reason: news jump +0.31 on cloud print.",
    order_id: "SIM-88421",
  },
  {
    time: ago(38),
    ticker: "US.COST",
    side: "BUY",
    qty: 40,
    price: 868.15,
    reason: "Action: Add Costco. Reason: US defensive bid, news +0.41.",
    order_id: "SIM-88410",
  },
  {
    time: ago(95),
    ticker: "HK.03690",
    side: "SELL",
    qty: 300,
    price: 118.4,
    reason: "Action: Trim Meituan. Reason: subsidy headline, score −0.31.",
    order_id: "SIM-88390",
  },
  {
    time: ago(140),
    ticker: "HK.03750",
    side: "BUY",
    qty: 500,
    price: 412.8,
    reason: "Action: Buy CATL. Reason: storage-order print.",
    order_id: "SIM-88371",
  },
  {
    time: ago(400),
    ticker: "US.KO",
    side: "SELL",
    qty: 100,
    price: 68.92,
    reason: "Action: Reduce KO. Reason: volume-only-okay China note.",
    order_id: "SIM-88202",
  },
  {
    time: ago(980),
    ticker: "HK.00700",
    side: "BUY",
    qty: 100,
    price: 408.6,
    reason: "Action: Seed Tencent on the open.",
    order_id: "SIM-87011",
  },
];

function buildSnapshot(): Snapshot {
  const equity = 1_012_440;
  const cash = 642_880;
  return {
    kind: "test",
    updated_at: ago(0.4),
    cash,
    equity,
    holdings: {
      "HK.00700": 1200,
      "HK.03690": -400,
      "HK.03750": 800,
      "US.COST": 90,
      "US.KO": 250,
    },
    last_action: {
      "HK.00700": 0.42,
      "HK.03690": -0.28,
      "HK.03750": 0.15,
      "US.COST": 0.33,
      "US.KO": -0.06,
    },
    last_reason: "TEST overlay — not a live book.",
    last_bar_datetime: "2026-08-23 11:30:00",
    news_scores: {
      "HK.00700": 0.31,
      "HK.03690": -0.22,
      "HK.03750": 0.08,
      "US.COST": 0.41,
      "US.KO": 0.04,
    },
    headlines: {
      "HK.00700": TENCENT,
      "HK.03690": MEITUAN,
      "HK.03750": CATL,
      "US.COST": COSTCO,
      "US.KO": KO,
    },
    fills: FILLS,
    initial_cash: 1_000_000,
    pnl: equity - 1_000_000,
  };
}

export function makeTestSnapshot(): SnapshotResponse {
  const latest = buildSnapshot();
  const equitySeries = Array.from({ length: 24 }, (_, i) => {
    const step = 23 - i;
    const wobble = Math.sin(i / 3) * 1800 + i * 420;
    return { t: ago(step * 8), equity: 998_200 + wobble };
  });
  return {
    latest,
    equitySeries,
    stale: false,
    staleAfterSeconds: 180,
  };
}
