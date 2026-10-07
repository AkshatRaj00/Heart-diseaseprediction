import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        background: "#030712",
        card: "#0f172a"
      }
    }
  },
  plugins: []
} satisfies Config;
