import { Dashboard } from "@/components/Dashboard";
import { fetchSnapshots } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const initial = await fetchSnapshots();
  return <Dashboard initial={initial} />;
}
