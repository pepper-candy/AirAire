"use client";

import { useEffect, useRef, useState } from "react";

const GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&*";
const SCRAMBLE_MS = 3000;
const HOLD_MS = 3000;

function randomGlyph(): string {
  return GLYPHS[Math.floor(Math.random() * GLYPHS.length)] || "X";
}

export function LoginForm({ wrongInitially }: { wrongInitially: boolean }) {
  const [wrong, setWrong] = useState(wrongInitially);
  const [shake, setShake] = useState(0);
  const [opening, setOpening] = useState(false);
  const [gate, setGate] = useState("");
  const [welcome, setWelcome] = useState("Welcome");
  const [decode, setDecode] = useState<{ chars: string[]; locked: number }>({
    chars: [],
    locked: 0,
  });
  const gateRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!opening) {
      return;
    }
    const letters = Array.from(welcome);
    const started = performance.now();
    let frame = 0;
    const scramble = (now: number) => {
      const elapsed = now - started;
      if (elapsed >= SCRAMBLE_MS) {
        setDecode({ locked: letters.length, chars: letters });
        return;
      }
      const locked = Math.floor((elapsed / SCRAMBLE_MS) * letters.length);
      setDecode({
        locked,
        chars: letters.map((ch, i) => (i < locked ? ch : randomGlyph())),
      });
      frame = window.requestAnimationFrame(scramble);
    };
    frame = window.requestAnimationFrame(scramble);
    const lockId = window.setTimeout(() => {
      window.cancelAnimationFrame(frame);
      setDecode({ locked: letters.length, chars: letters });
    }, SCRAMBLE_MS);
    const goId = window.setTimeout(() => {
      window.location.assign("/");
    }, SCRAMBLE_MS + HOLD_MS);
    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(lockId);
      window.clearTimeout(goId);
    };
  }, [opening, welcome]);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (opening) {
      return;
    }
    const form = event.currentTarget;
    const body = new FormData(form);
    let nextWelcome = "WELCOME";
    let ok = false;
    try {
      const res = await fetch("/api/gate", {
        method: "POST",
        body,
        headers: { Accept: "application/json" },
      });
      if (res.ok) {
        const data = (await res.json()) as { ok?: boolean; welcome?: string };
        ok = Boolean(data.ok);
        if (data.welcome) {
          nextWelcome = data.welcome;
        }
      }
    } catch {
      ok = false;
    }
    if (!ok) {
      setWrong(true);
      setShake((n) => n + 1);
      gateRef.current?.focus();
      return;
    }
    setWrong(false);
    setWelcome(nextWelcome);
    setOpening(true);
    setDecode({
      locked: 0,
      chars: Array.from(nextWelcome).map(() => randomGlyph()),
    });
  }

  return (
    <div className="login-stack">
      <form className="login-card" method="post" action="/api/gate" onSubmit={onSubmit}>
        <h1>Paper Book</h1>
        <div key={shake} className={wrong && !opening ? "login-row is-wrong" : "login-row"}>
          {opening ? (
            <div className="gate-decode" aria-live="polite" aria-label={welcome}>
              {decode.chars.map((ch, i) => (
                <span key={i} className={i < decode.locked ? "is-locked" : "is-scramble"}>
                  {ch === " " ? "\u00a0" : ch}
                </span>
              ))}
            </div>
          ) : (
            <input
              ref={gateRef}
              type="password"
              name="gate"
              autoFocus
              required
              placeholder="SECURITY_CODE"
              value={gate}
              onChange={(event) => setGate(event.target.value)}
              className={wrong ? "gate-wrong" : undefined}
              aria-invalid={wrong ? true : undefined}
            />
          )}
        </div>
      </form>
      <div className="eyebrow">Created by AirAire</div>
    </div>
  );
}
