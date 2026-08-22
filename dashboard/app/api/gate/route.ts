import { NextResponse } from "next/server";
import { expectedGate, GATE_COOKIE, gatesMatch } from "@/lib/gate";

function setGateCookie(res: NextResponse, value: string): NextResponse {
  res.cookies.set({
    name: GATE_COOKIE,
    value,
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
  });
  return res;
}

export async function POST(request: Request) {
  const expected = expectedGate();
  const form = await request.formData();
  const provided = String(form.get("gate") || "");
  const wantsJson = (request.headers.get("accept") || "").includes("application/json");

  if (!expected || !gatesMatch(provided, expected)) {
    if (wantsJson) {
      return NextResponse.json({ ok: false }, { status: 401 });
    }
    return NextResponse.redirect(new URL("/login?error=1", request.url), 303);
  }

  if (wantsJson) {
    return setGateCookie(NextResponse.json({ ok: true }), expected);
  }

  return setGateCookie(NextResponse.redirect(new URL("/", request.url), 303), expected);
}
