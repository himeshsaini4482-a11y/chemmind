/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        chem: {
          dark: "#0a0f0d",
          panel: "#111a16",
          border: "#1a2e24",
          primary: "#10b981",
          primaryHover: "#059669",
          primaryDark: "#064e3b",
          userBubble: "#064e3b",
          aiBubble: "#1a2e24",
          text: "#e2e8f0",
          textMuted: "#94a3b8",
        },
      },
    },
  },
  plugins: [],
};
