export const GATE_COOKIE = "dashboard_gate";

export type GateEntry = {
  password: string;
  welcome: string;
};

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

export function listGates(): GateEntry[] {
  const rows: GateEntry[] = [
    { password: (process.env.DASHBOARD_GATE_VICTOR || "Banana").trim(), welcome: "WELCOME, VICTOR🍌" },
    { password: (process.env.DASHBOARD_GATE_DEREK || "Whale").trim(), welcome: "WELCOME, DEREK🐳" },
    { password: (process.env.DASHBOARD_GATE_ADRIAN || "Star").trim(), welcome: "WELCOME, ADRIAN✨" },
    { password: (process.env.DASHBOARD_GATE_PUBLIC || "Public").trim(), welcome: "WELCOME" },
  ].filter((row) => row.password);

  const legacy = (process.env.DASHBOARD_GATE || "").trim();
  if (legacy && !rows.some((row) => gatesMatch(legacy, row.password))) {
    rows.push({ password: legacy, welcome: "WELCOME" });
  }
  return rows;
}

export function matchGate(provided: string | undefined | null): GateEntry | null {
  if (!provided) {
    return null;
  }
  for (const gate of listGates()) {
    if (gatesMatch(provided, gate.password)) {
      return gate;
    }
  }
  return null;
}
