"use client";

import { Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

export function ThemeToggle() {
  const [dark, setDark] = useState<boolean | null>(null);

  useEffect(() => {
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  function toggle() {
    const next = !document.documentElement.classList.contains("dark");
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("rag-theme", next ? "dark" : "light");
    setDark(next);
  }

  // Avoid flash by rendering only after we know the current state.
  if (dark === null) {
    return <div aria-hidden className="h-9 w-9" />;
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      className="bg-elev hover:border-strong inline-flex h-9 w-9 items-center justify-center rounded-lg border transition-colors"
    >
      {dark ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}
