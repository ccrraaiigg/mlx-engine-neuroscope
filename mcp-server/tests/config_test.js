/**
 * Configuration tests
 */

import { assertEquals, assertThrows } from '@std/assert';
import { validateConfig } from '../src/types/config.js';
import { defaultConfig } from '../src/config/defaults.js';

Deno.test('validateConfig - should accept valid configuration', () => {
  const result = validateConfig(defaultConfig);
  assertEquals(typeof result, 'object');
  assertEquals(result.mcp.port, 3000);
});

Deno.test('validateConfig - should reject invalid port', () => {
  const invalidConfig = {
    ...defaultConfig,
    mcp: { ...defaultConfig.mcp, port: -1 },
  };
  
  assertThrows(() => validateConfig(invalidConfig));
});