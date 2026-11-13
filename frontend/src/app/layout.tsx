
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Streamline | Data Intelligence Engine",
  description:
    "Discover topics, track news trends, detect drift, and analyze influential articles.",
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
