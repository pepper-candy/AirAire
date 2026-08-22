import { NextResponse } from "next/server";
import { fetchSnapshots } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET() {
  const data = await fetchSnapshots();
  return NextResponse.json(data);
}
