export const GATE_COOKIE = "dashboard_gate";

export function expectedGate(): string {
  return (process.env.DASHBOARD_GATE || "").trim();
}

export function gatesMatch(provided: string | undefined | null, expected: string): boolean {
  if (!expected || !provided || provided.length !== expected.length) {
    return false;
  }
  let mismatch = 0;
  for (let i = 0; i < expected.length; i += 1) {
    mismatch |= provided.charCodeAt(i) ^ expected.charCodeAt(i);
  }
  return mismatch === 0;
}
