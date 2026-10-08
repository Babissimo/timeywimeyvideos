import { defineConfig } from "vitest/config";
import { playwright } from "@vitest/browser-playwright";

// Fixtures in web/test/fixtures/: text via a `?raw` import in both projects; binary via fs
// (`new URL(path, import.meta.url)`) in Node and a `?url` import plus fetch in the browser.
export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: "unit",
          environment: "node",
          include: ["web/src/**/*.test.ts"],
          exclude: ["web/src/**/*.browser.test.ts"],
        },
      },
      {
        test: {
          name: "browser",
          include: ["web/src/**/*.browser.test.ts"],
          browser: {
            enabled: true,
            headless: true,
            // Needs Google Chrome installed: Playwright's bundled Chromium lacks H.264.
            provider: playwright({ launchOptions: { channel: "chrome" } }),
            instances: [{ browser: "chromium" }],
          },
        },
      },
    ],
  },
});
