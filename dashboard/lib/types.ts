export const CORE_TICKERS = ["HK.00700", "HK.03690", "HK.03750", "US.COST", "US.KO"] as const;

export const HK_TICKERS = ["HK.00700", "HK.03690", "HK.03750"] as const;
export const US_TICKERS = ["US.COST", "US.KO"] as const;
export const HK_INITIAL_CASH = 1_000_000;
export const US_INITIAL_CASH = 1_000_000;

/** Unfilled CATL sells that must not count as fills (400 @ 638 + 700 @ 637.50). */
export const PHANTOM_CATL_SELLS = [
  { order_id: "8899494", ticker: "HK.03750", qty: 400, price: 638.0 },
  { order_id: "8899530", ticker: "HK.03750", qty: 700, price: 637.5 },
] as const;

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
  /** BUY/SELL while side is PENDING (working OpenD limit). */
  working_side?: string;
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
  /** Last live marks from OpenD when the VM pushed them. Optional on old rows. */
  prices?: Record<string, number>;
  /** US SIMULATE cash (USD). Separate from HK HKD cash. Optional on old rows. */
  us_cash?: number;
  us_equity?: number;
  /** OpenD working limits — matching fills must display as PENDING, not BUY/SELL. */
  pending_order_ids?: string[];
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

export type RangeMode = "today" | "week";

export type EquityMeta = {
  range: RangeMode;
  day: string;
  rawCount: number;
  shownCount: number;
  bucketMinutes: number;
  hkRawCount?: number;
  usRawCount?: number;
  hkShownCount?: number;
  usShownCount?: number;
};

export type SnapshotResponse = {
  latest: Snapshot | null;
  /** @deprecated mashed HKD+USD path — use hkEquitySeries / usEquitySeries */
  equitySeries: EquityPoint[];
  hkEquitySeries: EquityPoint[];
  usEquitySeries: EquityPoint[];
  equityMeta?: EquityMeta;
  stale: boolean;
  staleAfterSeconds: number;
  error?: string;
};

export const STALE_AFTER_SECONDS = 180;
