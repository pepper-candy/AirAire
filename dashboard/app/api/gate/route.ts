import { NextResponse } from "next/server";
import { expectedGate, GATE_COOKIE, gatesMatch } from "@/lib/gate";

export async function POST(request: Request) {
  const expected = expectedGate();
  const form = await request.formData();
  const provided = String(form.get("gate") || "");

  if (!expected || !gatesMatch(provided, expected)) {
    return NextResponse.redirect(new URL("/login?error=1", request.url), 303);
  }

  const res = NextResponse.redirect(new URL("/", request.url), 303);
  res.cookies.set({
    name: GATE_COOKIE,
    value: expected,
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
  });
  return res;
}
