/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ["JetBrains Mono", "Cascadia Mono", "Consolas", "monospace"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 28px rgba(34, 211, 238, 0.18)",
        alert: "0 0 28px rgba(248, 113, 113, 0.18)",
      },
    },
  },
  plugins: [],
};
