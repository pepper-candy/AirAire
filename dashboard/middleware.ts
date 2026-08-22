import { NextResponse, type NextRequest } from "next/server";
import { expectedGate, GATE_COOKIE, gatesMatch } from "./lib/gate";

export function middleware(request: NextRequest) {
  const expected = expectedGate();
  const { pathname } = request.nextUrl;

  if (pathname.startsWith("/login") || pathname.startsWith("/api/gate")) {
    return NextResponse.next();
  }

  if (!expected) {
    return NextResponse.next();
  }

  const cookie = request.cookies.get(GATE_COOKIE)?.value;
  if (gatesMatch(cookie, expected)) {
    return NextResponse.next();
  }

  const queryGate = request.nextUrl.searchParams.get("gate");
  if (gatesMatch(queryGate, expected)) {
    const clean = request.nextUrl.clone();
    clean.searchParams.delete("gate");
    const redirectTo = clean.pathname === "/login" ? "/" : clean.pathname + clean.search;
    const res = NextResponse.redirect(new URL(redirectTo, request.url));
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

  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "gated" }, { status: 401 });
  }

  return NextResponse.redirect(new URL("/login", request.url));
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
