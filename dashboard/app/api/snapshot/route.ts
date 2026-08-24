import { NextResponse } from "next/server";
import { fetchLatest } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET() {
  const data = await fetchLatest();
  return NextResponse.json(data);
}
