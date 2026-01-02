import type { Metadata } from "next";
import localFont from "next/font/local";
import "computer-modern/cmu-classical-serif.css";
import "./globals.css";

const suit = localFont({
  src: "../../public/fonts/SUIT-Variable.woff2",
  variable: "--font-suit",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Yesterday's close, Today's edge",
  description: "매일 아침, AI가 전날 미국 증시의 주요 흐름과 핵심 종목 동향을 분석해 팟캐스트로 전달합니다.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body className={`${suit.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
