-- Rebuild the latest blotter from the fill log (qty × price).
-- Does NOT copy OpenD / Futu handling fees.
--
-- Phantom CATL sells that were booked as fills while still pending:
--   8899494  400 @ 638.00
--   8899530  700 @ 637.50
-- Those rows become CANCEL and do not move cash or qty.
--
-- Paste in Supabase SQL editor (table owner). Dashboard KPIs also ignore
-- mashed snapshot cash and recompute from fills, so a 60s trader push will
-- not put Futu-fee numbers back on screen. To stop the pickle from sending
-- the old cash, run on the GPU box (trader stopped):
--   python -m src.correct_unfilled --from-fills
-- then restart V3. Do not run reconcile_futu --apply.

DO $$
DECLARE
  rec record;
  p jsonb;
  fill jsonb;
  new_fills jsonb := '[]'::jsonb;
  side text;
  ticker text;
  oid text;
  qty numeric;
  px numeric;
  hk_cash numeric := 1000000;
  us_cash numeric := 1000000;
  holdings jsonb := '{
    "HK.00700": 0,
    "HK.03690": 0,
    "HK.03750": 0,
    "US.COST": 0,
    "US.KO": 0
  }'::jsonb;
  marks jsonb := '{}'::jsonb;
  hk_mtm numeric := 0;
  us_mtm numeric := 0;
  hk_equity numeric;
  us_equity numeric;
  initial_cash numeric;
  t text;
BEGIN
  SELECT id, payload INTO rec
  FROM public.bot_snapshots
  ORDER BY created_at DESC
  LIMIT 1;

  IF rec.id IS NULL THEN
    RAISE NOTICE 'No bot_snapshots row.';
    RETURN;
  END IF;

  p := rec.payload;
  initial_cash := COALESCE((p->>'initial_cash')::numeric, 1000000);

  FOR fill IN
    SELECT value
    FROM jsonb_array_elements(COALESCE(p->'fills', '[]'::jsonb))
  LOOP
    oid := COALESCE(fill->>'order_id', '');
    IF oid IN ('8899494', '8899530') THEN
      fill := jsonb_set(fill, '{side}', '"CANCEL"');
      fill := jsonb_set(
        fill,
        '{reason}',
        to_jsonb(
          'CANCEL unfilled limit — shares restored. ' || COALESCE(fill->>'reason', '')
        )
      );
    END IF;
    new_fills := new_fills || jsonb_build_array(fill);

    ticker := COALESCE(fill->>'ticker', '');
    px := COALESCE(NULLIF(fill->>'price', '')::numeric, 0);
    IF ticker <> '' AND px > 0 THEN
      marks := marks || jsonb_build_object(ticker, px);
    END IF;

    side := upper(COALESCE(fill->>'side', ''));
    IF side NOT IN ('BUY', 'SELL') THEN
      CONTINUE;
    END IF;
    qty := COALESCE(NULLIF(fill->>'qty', '')::numeric, 0);
    IF ticker = '' OR qty <= 0 OR px <= 0 THEN
      CONTINUE;
    END IF;

    holdings := holdings || jsonb_build_object(
      ticker,
      COALESCE((holdings->>ticker)::numeric, 0) + CASE WHEN side = 'BUY' THEN qty ELSE -qty END
    );

    IF left(ticker, 3) = 'US.' THEN
      us_cash := us_cash + CASE WHEN side = 'BUY' THEN -qty * px ELSE qty * px END;
    ELSE
      hk_cash := hk_cash + CASE WHEN side = 'BUY' THEN -qty * px ELSE qty * px END;
    END IF;
  END LOOP;

  FOREACH t IN ARRAY ARRAY['HK.00700', 'HK.03690', 'HK.03750'] LOOP
    hk_mtm := hk_mtm + COALESCE((holdings->>t)::numeric, 0) * COALESCE((marks->>t)::numeric, 0);
  END LOOP;
  FOREACH t IN ARRAY ARRAY['US.COST', 'US.KO'] LOOP
    us_mtm := us_mtm + COALESCE((holdings->>t)::numeric, 0) * COALESCE((marks->>t)::numeric, 0);
  END LOOP;

  hk_equity := hk_cash + hk_mtm;
  us_equity := us_cash + us_mtm;

  p := p
    || jsonb_build_object(
      'kind', 'live',
      'updated_at', now(),
      'cash', hk_cash,
      'equity', hk_equity,
      'pnl', hk_equity - initial_cash,
      'initial_cash', initial_cash,
      'us_cash', us_cash,
      'us_equity', us_equity,
      'holdings', holdings,
      'fills', new_fills,
      'prices', marks,
      'last_reason',
        'Fill-log rebuild: CATL 8899494/8899530 are CANCEL. Cash is qty×price, no Futu fees. HK/US are separate books.'
    );

  INSERT INTO public.bot_snapshots (kind, payload)
  VALUES ('live', p);

  RAISE NOTICE 'Inserted fill-log book HK cash=% equity=% CATL=% US cash=%',
    hk_cash,
    hk_equity,
    COALESCE((holdings->>'HK.03750')::numeric, 0),
    us_cash;
END $$;
