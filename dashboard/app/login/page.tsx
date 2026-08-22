import { LoginForm } from "@/components/LoginForm";

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  const params = await searchParams;
  return (
    <div className="login-wrap">
      <LoginForm wrongInitially={Boolean(params.error)} />
    </div>
  );
}
