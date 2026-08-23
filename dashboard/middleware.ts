import { NextResponse, type NextRequest } from "next/server";
import { GATE_COOKIE, listGates, matchGate } from "./lib/gate";

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (pathname.startsWith("/login") || pathname.startsWith("/api/gate")) {
    return NextResponse.next();
  }

  if (listGates().length === 0) {
    return NextResponse.next();
  }

  const cookie = request.cookies.get(GATE_COOKIE)?.value;
  if (matchGate(cookie)) {
    return NextResponse.next();
  }

  const queryGate = request.nextUrl.searchParams.get("gate");
  const queried = matchGate(queryGate);
  if (queried) {
    const clean = request.nextUrl.clone();
    clean.searchParams.delete("gate");
    const redirectTo = clean.pathname === "/login" ? "/" : clean.pathname + clean.search;
    const res = NextResponse.redirect(new URL(redirectTo, request.url));
    res.cookies.set({
      name: GATE_COOKIE,
      value: queried.password,
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
