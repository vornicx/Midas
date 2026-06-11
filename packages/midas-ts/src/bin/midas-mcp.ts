#!/usr/bin/env node
import { main } from "../mcp.js";

main().catch((err) => {
  console.error("[midas-mcp-ts] fatal:", err);
  process.exit(1);
});
