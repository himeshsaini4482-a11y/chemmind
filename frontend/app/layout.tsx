import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ChemMind - AI for Computational Chemistry",
  description: "Multi-agent AI system for computational chemistry PhD students",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
