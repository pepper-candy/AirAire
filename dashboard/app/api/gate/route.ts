import { NextResponse } from "next/server";
import { GATE_COOKIE, listGates, matchGate } from "@/lib/gate";

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
  const form = await request.formData();
  const provided = String(form.get("gate") || "");
  const wantsJson = (request.headers.get("accept") || "").includes("application/json");
  const matched = listGates().length > 0 ? matchGate(provided) : null;

  if (!matched) {
    if (wantsJson) {
      return NextResponse.json({ ok: false }, { status: 401 });
    }
    return NextResponse.redirect(new URL("/login?error=1", request.url), 303);
  }

  if (wantsJson) {
    return setGateCookie(NextResponse.json({ ok: true, welcome: matched.welcome }), matched.password);
  }

  return setGateCookie(NextResponse.redirect(new URL("/", request.url), 303), matched.password);
}
