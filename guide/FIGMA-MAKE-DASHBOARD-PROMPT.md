# Figma Make prompt — AirAire paper blotter (dark)

Paste everything below the line into **Figma Make**. Do not design the login screen. After you like the frames, import them with MCP / design-to-code into `dashboard/components/Dashboard.tsx` only.

---

You are Figma Make. Design a **production-quality dark-mode trading blotter** for a student paper-trading project called **AirAire**. This is a **read-only** website. No order tickets, no buy/sell buttons, no settings, no user avatar, no chat.

Do **not** design a login page. Login already exists and must stay as-is.

## Product (read this, do not invent features)

Two university students watch a PPO bot that paper-trades 5 names on a GPU VM (Futu SIMULATE). The VM dies when the campus VDI idles. The site stays up 24/7 and shows the **last snapshot** from Supabase.

The page polls every 20 seconds. Humans need **headlines**. The model only uses a **sentiment number**. Telegram already sends fill text; this page is the shared visual book.

**Out of scope:** live ticks, candlesticks, depth, friend placing orders, light mode, marketing landing page, logo illustration of airplanes, purple “AI gradient” chrome, glassmorphism, 3D, stock photos.

## Screens to produce

1. **Desktop blotter 1440×1024** — primary. Named `Blotter / Desktop / Live`.
2. **Desktop blotter stale** — same layout. Named `Blotter / Desktop / Stale`.
3. **Desktop blotter empty** — first visit, no snapshot yet. Named `Blotter / Desktop / Empty`.
4. **Mobile blotter 390×844** — stacked. Named `Blotter / Mobile / Live`.

Optional: one **component set** for KPI card, ticker row, fill row, headline row (default / hover / empty).

Auto-layout everywhere. 8px grid. No overlapping text. No tiny grey-on-grey.

## Visual direction

Dark **terminal / night desk**, not crypto-casino and not generic SaaS.

- Background: near-black warm ink `#0B0C0F` to `#12141A`
- Surfaces: `#171A21`, hairline `#2A2F3A`
- Text: `#E8E6E1` primary, `#9AA3B2` muted
- Live / ok: muted sage `#8FBF9F`
- Stale / warning: amber `#D4A574` (banner only, not the whole page)
- Positive numbers: `#7DCE8A`
- Negative numbers: `#E07A6A`
- Neutral numbers: `#E8E6E1`

Type: **IBM Plex Sans** for UI, **IBM Plex Mono** for money, scores, times, tickers. Tabular lining figures. No Inter, no Poppins, no Comic rounded.

Radius 6–8px max. Dense but breathable. Feels like a **paper book you can read on a phone at lunch**.

Brand wordmark: small eyebrow `AIRAIRE` + title `Paper book`. Subtitle `Read-only · the VM trades`.

## Information architecture (must all appear)

### A. Status strip (full width, top)

Two variants:

- **Live:** sage left pip. Copy: `VM snapshot is fresh. This page does not trade.`
- **Stale:** amber strip. Copy: `VM idle or unreachable. Showing last snapshot. Fresh for 180s.`

Right side of strip: `Last snapshot  22 Aug 2026, 13:40:01 HKT` and `Last bar  2026-08-22 13:30:00`.

### B. KPI row (4 cards)

Use these exact sample numbers (paper account, start cash 1,000,000):

| Label | Value | Note |
|---|---|---|
| Equity | 1,012,440.00 | mono, large |
| Cash | 486,210.50 | mono |
| P&L vs start | +12,440.00 | green if ≥ 0, red if < 0 |
| Equity path | small sparkline | 24 points, slight upward drift, sage stroke, no grid, no axis labels |

Empty state: same cards with `—` and sparkline placeholder `Waiting for snapshots`.

### C. Main split (desktop 1.2fr / 0.8fr; mobile stack)

**Left — Book**

One-line last reason (muted):  
`Action: Buy 100 Tencent. Reason: News Sentiment jumped + High negative correlation detected.`

Table, 5 rows only, these names:

| Display | Code | Holdings | Action | News |
|---|---|---|---|---|
| Tencent | HK.00700 | 1,200 | +0.412 | +0.31 |
| Meituan | HK.03690 | 0 | −0.020 | −0.12 |
| CATL | HK.03750 | 800 | +0.180 | +0.08 |
| Costco | US.COST | 15 | −0.310 | −0.44 |
| Coca-Cola | US.KO | 40 | +0.055 | +0.02 |

Action and News are policy/sentiment in **[-1, 1]**, 3 decimals, color by sign (threshold ±0.05). Holdings are shares, not dollars. Code in mono under the name.

**Right — Fills (blotter)**

Newest first. 4 sample rows:

- `13:31:02 HKT` · **BUY** 100 Tencent @ 412.20 · reason one line muted
- `13:20:11 HKT` · **SELL** 200 Meituan @ 98.40
- `10:45:00 HKT` · **BUY** 5 Costco @ 912.15
- `09:41:08 HKT` · **BUY** 800 CATL @ 312.00

BUY tag sage, SELL tag rust. Empty: `No SIMULATE fills yet.`

### D. Headlines the model scored (full width below)

Five groups, one per ticker. Group header: name + news score (same color rule). Up to **5–8** headlines each. Each row:

- Title (can wrap 2 lines)
- Meta mono: `Reuters · 20260822T043000 · +0.21`
- Title is a text-link affordance (underline on hover). No favicons.

Sample Tencent titles (invent 3 more in the same tone):

- `Tencent cloud unit posts stronger-than-expected quarter`
- `HK tech slips as US futures firm; 0700 in focus`

Empty group: `No cached headlines for this name.`

## Layout rules for later code import (MCP)

- One top-level frame per screen. Names exactly as above.
- Nested Auto Layout: `Page > Status > Header > KPIs > Split(Book, Fills) > Headlines`.
- Do not put Book and Headlines in one scrolling mystery stack without section titles.
- All repeating rows must be **components** with consistent padding so a developer can map 1:1 to React.
- Do not add filters, date pickers, search, tabs for “Analytics”, or a sidebar nav.
- Do not show HSI / SPX. Only the five names above.
- Do not show account login chrome on this page.

## Copy tone

Calm, precise, slightly dry. No “Welcome back!”, no “Let’s make money”, no emojis, no rockets.

Header right meta stays Hong Kong time, 24-hour.

## Deliverable

High-fidelity dark UI, ready for design-to-code. Desktop Live is the hero. Stale and Empty must be clearly different **only** in the status strip and placeholders — do not redesign the grid between states.
