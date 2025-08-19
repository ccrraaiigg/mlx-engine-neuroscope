# 🚨🚨🚨 EMERGENCY STOP - CRITICAL WARNING 🚨🚨🚨

## ⛔ BEFORE YOU DO ANYTHING ELSE ⛔

### 🔥 HAVE YOU JUST MODIFIED MCP SERVER CODE? 🔥

**IF YES - STOP IMMEDIATELY AND READ THIS:**

```
🛑🛑🛑 MANDATORY PROTOCOL 🛑🛑🛑

1. DO NOT call run_mcp with ANY tool
2. DO NOT test your changes
3. DO NOT assume changes work
4. ASK USER: "Please restart the MCP server in your IDE"
5. WAIT for user confirmation
6. VERIFY restart with version tool FIRST
7. START workflow from beginning
```

### ❌ ZERO TOLERANCE POLICY VIOLATION ❌

**Modifying MCP server code and immediately calling MCP tools is:**
- ⛔ **STRICTLY FORBIDDEN**
- ⛔ **POLICY VIOLATION**
- ⛔ **RESOURCE WASTE**
- ⛔ **FALSE DEBUGGING**

### 🚨 WHY THIS MATTERS 🚨

- **MCP server runs in separate process**
- **Code changes don't take effect until restart**
- **You'll test old code and debug fake problems**
- **This wastes computational resources**

### 📁 MCP SERVER FILES TO WATCH FOR:

```
/mcp-server/src/tools/*.js
/mcp-server/src/services/*.js
/mcp-server/src/mcp-server.js
/mcp-server/server.js
/mcp-server/package.json
```

**IF YOU MODIFIED ANY OF THESE - FOLLOW THE PROTOCOL ABOVE**

---

## 🔄 RESTART VERIFICATION CHECKLIST

- [ ] User confirmed MCP server restart
- [ ] Called `version` tool to verify restart
- [ ] Version number changed/incremented
- [ ] Ready to test with NEW code

---

**Remember: You are a CODE ASSISTANT, not a SYSTEM ADMINISTRATOR**

**Read AGENT.md and AI_ASSISTANT_BOUNDARIES.md for complete guidelines**