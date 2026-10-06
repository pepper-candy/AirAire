import { Dashboard } from "@/components/Dashboard";
import { fetchSnapshots } from "@/lib/supabase";
import { makeTestSnapshot } from "@/lib/testSnapshot";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  try {
    const initial = await fetchSnapshots();
    if (!initial.error) return <Dashboard initial={initial} />;
  } catch {
    // Supabase paused or unreachable
  }
  return <Dashboard initial={makeTestSnapshot()} startDemo />;
}
