"use client";

import { BarChart3, ClipboardList, Database, Library, Sparkles } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const TABS = [
  { href: "/", label: "Ask", Icon: Sparkles },
  { href: "/corpus", label: "Corpus", Icon: Library },
  { href: "/eval", label: "Eval", Icon: ClipboardList },
  { href: "/metrics", label: "Metrics", Icon: BarChart3 },
  { href: "/ingest", label: "Ingest", Icon: Database },
] as const;

export function NavTabs() {
  const pathname = usePathname();
  return (
    <nav className="flex items-center gap-1" aria-label="Primary">
      {TABS.map(({ href, label, Icon }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            className={cn(
              "inline-flex h-9 items-center gap-1.5 rounded-md px-3 text-sm font-medium transition-colors",
              active
                ? "bg-sunken border-strong border"
                : "text-muted border-transparent hover:text-[var(--color-fg)]",
            )}
          >
            <Icon size={14} />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
