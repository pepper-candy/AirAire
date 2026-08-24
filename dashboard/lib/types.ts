export const CORE_TICKERS = ["HK.00700", "HK.03690", "HK.03750", "US.COST", "US.KO"] as const;

export const TICKER_NAMES: Record<string, string> = {
  "HK.00700": "Tencent",
  "HK.03690": "Meituan",
  "HK.03750": "CATL",
  "US.COST": "Costco",
  "US.KO": "Coca-Cola",
};

export type Headline = {
  title: string;
  source: string;
  url: string;
  time_published: string;
  sentiment_score: number;
};

export type HeadlineBasket = {
  id: string;
  title: string;
  members: string[];
  headlines: Headline[];
};

export type Fill = {
  time: string;
  ticker: string;
  side: string;
  qty: number;
  price: number;
  reason: string;
  order_id: string;
};

export type Snapshot = {
  kind: string;
  updated_at: string;
  cash: number;
  equity: number;
  holdings: Record<string, number>;
  last_action: Record<string, number>;
  last_reason: string;
  last_bar_datetime: string;
  news_scores: Record<string, number>;
  headlines: Record<string, Headline[]>;
  headline_baskets?: HeadlineBasket[];
  fills: Fill[];
  initial_cash: number;
  pnl: number;
};

export type SnapshotRow = {
  id: number;
  created_at: string;
  kind: string;
  payload: Snapshot;
};

export type EquityPoint = {
  t: string;
  equity: number;
};

export type RangeMode = "today" | "week" | "day";

export type EquityMeta = {
  range: RangeMode;
  day: string;
  rawCount: number;
  shownCount: number;
  bucketMinutes: number;
};

export type SnapshotResponse = {
  latest: Snapshot | null;
  equitySeries: EquityPoint[];
  equityMeta?: EquityMeta;
  stale: boolean;
  staleAfterSeconds: number;
  error?: string;
};

export const STALE_AFTER_SECONDS = 180;
