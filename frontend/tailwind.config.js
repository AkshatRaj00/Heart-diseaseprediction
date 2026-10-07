/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primer: {
          bg: "#0d1117",
          card: "#161b22",
          border: "#30363d",
          hover: "#21262d",
          blue: "#58a6ff",
          green: "#3fb950",
          red: "#f85149",
          yellow: "#d29922",
          text: "#c9d1d9",
          muted: "#8b949e",
        }
      }
    },
  },
  plugins: [],
}
