import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CardioSense Enterprise | UCI Cleveland Clinical Intelligence",
  description: "Dual-Engine ML + PyTorch Deep Residual Clinical Decision Support System",
  viewport: "width=device-width, initial-scale=1, maximum-scale=1",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
      </head>
      <body className="min-h-screen bg-[#090d13] text-[#c9d1d9] antialiased selection:bg-[#58a6ff]/20">
        {children}
      </body>
    </html>
  );
}
