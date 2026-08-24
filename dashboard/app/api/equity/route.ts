import { NextRequest, NextResponse } from "next/server";
import { parseDayParam, parseRangeMode } from "@/lib/equityRange";
import { fetchEquitySeries } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const range = parseRangeMode(req.nextUrl.searchParams.get("range"));
  const day = parseDayParam(req.nextUrl.searchParams.get("day"));
  const data = await fetchEquitySeries(range, day);
  return NextResponse.json(data);
}
