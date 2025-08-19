# MCP Server Restart Automation & Reminder System

## CRITICAL REMINDER FOR AI ASSISTANTS

⚠️ **NEVER TEST MCP SERVER CODE CHANGES WITHOUT USER RESTART CONFIRMATION** ⚠️

### The Problem
AI assistants repeatedly forget that MCP server code changes require manual restart by the user in the IDE. Common mistakes include:
- Attempting to restart services directly with shell commands
- Trying to kill/check processes with `ps`, `kill`, `pkill`
- Reading PID files to manipulate running services
- Testing code changes without restart verification
- Assuming changes work because "they should"

This leads to:
- Wasted time testing with old code
- Frustration from repeated failures
- Inefficient debugging cycles
- Violation of AI assistant role boundaries

### Mandatory Protocol for AI Assistants

#### BEFORE Any MCP Server Testing:
1. **ALWAYS ASK**: "Have you restarted the MCP server since the last code change?"
2. **WAIT FOR CONFIRMATION**: Do not proceed without explicit user confirmation
3. **VERSION CHECK**: Use `run_mcp` with `version` tool to verify restart
4. **DOCUMENT**: Note the restart in your workflow

#### After Making Code Changes:
1. **IMMEDIATE STOP**: Do not attempt to test the changes
2. **CLEAR INSTRUCTION**: Tell user exactly what to restart
3. **WAIT STATE**: Mark testing tasks as "pending restart"
4. **REMINDER**: Reference this document

### Automated Reminder Triggers

#### Code Change Detection
If you modify any of these files, MANDATORY restart required:
- `mcp-server/src/tools/*.js`
- `mcp-server/src/services/*.js`
- `mcp-server/src/mcp-server.js`
- `mcp-server/server.js`
- `mcp-server/package.json`

#### Testing Prevention
BEFORE calling any MCP tools after code changes:
1. Check if restart confirmation received
2. If no confirmation, STOP and request restart
3. Use version check to verify

### 🚨 CRITICAL: Process Manipulation Prevention 🚨

**AI ASSISTANTS MUST NEVER**:
- Use `ps aux | grep` to check processes
- Use `kill`, `pkill`, `killall` to stop processes
- Read `.pid` files to manipulate services
- Use `systemctl`, `service`, or similar commands
- Attempt any direct process management

**ONLY USER CAN**:
- Restart services manually
- Manage system processes
- Control service lifecycle

**AI ROLE**: Code assistant only - NOT system administrator

### User Restart Instructions

```
🔄 MCP SERVER RESTART REQUIRED

You need to restart the MCP server in your IDE:
1. Stop the current MCP server process
2. Restart it through your IDE's MCP interface
3. Confirm restart completion
4. I will then verify with a version check

Do not proceed until restart is complete!
```

### Verification Protocol

```javascript
// Always run this after user confirms restart
run_mcp({
  server_name: "mcp.config.usrlocalmcp.mechanistic-interpretability",
  tool_name: "version",
  args: { random_string: "restart_verification" }
})
```

### Success Indicators
- Version number increments or changes
- Health check passes
- New functionality works as expected

### Failure Indicators
- Same version number
- Old behavior persists
- Error messages about missing functions

## Implementation Checklist

- [ ] Reference this document before any MCP testing
- [ ] Always ask for restart confirmation
- [ ] Use version check verification
- [ ] Document restart in workflow
- [ ] Never assume restart happened

---

**Remember**: The MCP server runs in a separate process. Code changes are not automatically reflected until manual restart!