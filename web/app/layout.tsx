import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geist = Geist({ subsets: ["latin"], variable: "--font-geist" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono" });

export const metadata: Metadata = {
  title: "RAG Engine",
  description:
    "Multi-agent RAG with hybrid retrieval, neural reranking, per-claim grounding, and an explicit insufficient-evidence exit.",
};

// Inline theme bootstrap. Runs before React hydrates so we never flash the
// wrong colour scheme. Stored under "rag-theme" so we don't collide with the
// site's other localStorage entries.
const themeBootstrap = `
(function() {
  try {
    var stored = localStorage.getItem("rag-theme");
    var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    var dark = stored ? stored === "dark" : prefersDark;
    if (dark) document.documentElement.classList.add("dark");
  } catch (_) {}
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geist.variable} ${geistMono.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeBootstrap }} />
      </head>
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
