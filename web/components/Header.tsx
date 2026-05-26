import { ShieldCheck } from "lucide-react";

import { NavTabs } from "./NavTabs";
import { ThemeToggle } from "./ThemeToggle";

export function Header() {
  return (
    <header className="bg-elev/80 sticky top-0 z-30 border-b backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex min-w-0 items-center gap-2">
          <span
            aria-hidden
            className="inline-flex h-7 w-7 items-center justify-center rounded-md"
            style={{
              background: "color-mix(in oklch, var(--color-accent), transparent 86%)",
              color: "var(--color-accent)",
            }}
          >
            <ShieldCheck size={15} strokeWidth={2.25} />
          </span>
          <div className="truncate text-sm font-medium">
            Multi-Agent RAG Intelligence Engine
          </div>
        </div>
        <div className="flex-1" />
        <NavTabs />
        <ThemeToggle />
      </div>
    </header>
  );
}
