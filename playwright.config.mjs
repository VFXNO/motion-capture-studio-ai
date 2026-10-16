import {defineConfig} from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  timeout: 30000,
  workers: 1,
  use: {baseURL: 'http://127.0.0.1:8766', browserName: 'chromium',
    viewport: {width: 1440, height: 1050}, screenshot: 'only-on-failure'},
  webServer: {command: 'python serve.py --port 8766', url: 'http://127.0.0.1:8766', reuseExistingServer: true}
});
