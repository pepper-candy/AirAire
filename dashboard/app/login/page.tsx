export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  const params = await searchParams;
  return (
    <div className="login-wrap">
      <form className="login-card" method="post" action="/api/gate">
        <div className="eyebrow">AirAire</div>
        <h1>Paper book</h1>
        <p>Read-only. The university VM is the only trader. Enter the shared gate.</p>
        <input type="password" name="gate" autoFocus required placeholder="DASHBOARD_GATE" />
        {params.error ? <p className="err">Wrong gate.</p> : null}
        <button type="submit">Open blotter</button>
      </form>
    </div>
  );
}
