/**
 * One-shot: restore unfilled CATL sells on the live blotter.
 * Reads DASHBOARD_PUSH_* from repo .env. Stop the V3 trader first.
 */
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const envText = fs.readFileSync(path.join(ROOT, ".env"), "utf8");
const env = {};
for (const line of envText.split(/\r?\n/)) {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("#")) continue;
  const eq = trimmed.indexOf("=");
  if (eq < 0) continue;
  env[trimmed.slice(0, eq).trim()] = trimmed.slice(eq + 1).trim();
}

const base = (env.DASHBOARD_PUSH_URL || "").replace(/\/$/, "");
const key = env.DASHBOARD_PUSH_KEY || "";
const rest = base.endsWith("/rest/v1") ? base : `${base}/rest/v1`;
const headers = {
  apikey: key,
  Authorization: `Bearer ${key}`,
  "Content-Type": "application/json",
  Prefer: "return=minimal",
};

const CANCELS = [
  { order_id: "8899494", ticker: "HK.03750", qty: 400, price: 638.0 },
  { order_id: "8899530", ticker: "HK.03750", qty: 700, price: 637.5 },
];
const CORE = ["HK.00700", "HK.03690", "HK.03750", "US.COST", "US.KO"];

async function main() {
  const getUrl = `${rest}/bot_snapshots?select=id,created_at,kind,payload&order=created_at.desc&limit=1`;
  const got = await fetch(getUrl, { headers: { apikey: key, Authorization: `Bearer ${key}` } });
  if (!got.ok) {
    throw new Error(`GET ${got.status} ${await got.text()}`);
  }
  const rows = await got.json();
  const payload = JSON.parse(JSON.stringify(rows[0].payload));
  const catl0 = Number(payload.holdings["HK.03750"] || 0);
  if (catl0 >= 1299) {
    console.log(`already restored CATL=${catl0}`);
    return;
  }
  const cash0 = Number(payload.cash);
  const equity0 = Number(payload.equity);
  const mark = catl0 > 0 ? (equity0 - cash0) / catl0 : 637;
  let cash = cash0;
  const holdings = {};
  for (const t of CORE) holdings[t] = Number(payload.holdings[t] || 0);
  const ids = new Set(CANCELS.map((r) => r.order_id));
  for (const row of CANCELS) {
    holdings[row.ticker] += row.qty;
    cash -= row.qty * row.price;
  }
  const equity = cash + holdings["HK.03750"] * mark;
  payload.cash = cash;
  payload.holdings = holdings;
  payload.equity = equity;
  payload.pnl = equity - Number(payload.initial_cash || 1e6);
  payload.kind = "live";
  payload.updated_at = new Date().toISOString();
  payload.last_reason = `Unfilled CATL sells cancelled (400 @ 638 + 700 @ 637.50). Book restored; holdings CATL=${holdings["HK.03750"]}.`;
  payload.fills = (payload.fills || []).map((f) => {
    if (!ids.has(String(f.order_id))) return f;
    return {
      ...f,
      side: "CANCEL",
      reason: `CANCEL unfilled limit — shares restored. ${f.reason || ""}`,
    };
  });
  const post = await fetch(`${rest}/bot_snapshots`, {
    method: "POST",
    headers,
    body: JSON.stringify({ kind: "live", payload }),
  });
  if (!post.ok) {
    throw new Error(`POST ${post.status} ${await post.text()}`);
  }
  console.log(`pushed CATL=${holdings["HK.03750"]} cash=${cash.toFixed(2)} equity=${equity.toFixed(2)}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
