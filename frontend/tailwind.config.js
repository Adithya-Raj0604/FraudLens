/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ["Fira Code", "ui-monospace", "monospace"],
        sans: ["Fira Sans", "system-ui", "sans-serif"],
      },
      colors: {
        base: "#020617",
        surface: "#0F172A",
        "surface-2": "#1E293B",
        accent: "#22C55E",
        danger: "#EF4444",
      },
    },
  },
  plugins: [],
}

