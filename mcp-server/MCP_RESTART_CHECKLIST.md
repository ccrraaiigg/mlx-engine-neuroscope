# MCP Server Restart Checklist

## 🔄 MANDATORY PROTOCOL FOR ALL CODE CHANGES

### Before Making Any Code Changes:
- [ ] Note current version number from last `version` tool call
- [ ] Plan changes to minimize restart cycles

### After Making Code Changes:
- [ ] **STOP ALL TESTING** - changes are not active yet!
- [ ] Increment version number in version tool
- [ ] Ask user: "Please restart the MCP server in your IDE for code changes to take effect"
- [ ] Wait for user confirmation of restart

### After User Restarts Server:
- [ ] **First tool call MUST be `version`** to verify restart
- [ ] Confirm version number matches your increment
- [ ] If version doesn't match → Ask user to restart again

### After Restart Verification:
- [ ] Load fresh model (old model was killed)
- [ ] Run health check
- [ ] Restart demo workflow from beginning
- [ ] Test new code behavior

## 🚫 NEVER DO THESE:
- ❌ Test code changes without restart verification
- ❌ Assume changes work because "they should"
- ❌ Debug issues that don't exist in new code
- ❌ Continue with old model/data after restart
- ❌ **Use shell commands to check/kill processes**
- ❌ **Read PID files to manipulate services**
- ❌ **Attempt to restart services directly**
- ❌ **Use `ps`, `kill`, `pkill`, `systemctl` commands**
- ❌ **Act as system administrator**

## ✅ SUCCESS INDICATORS:
- Version number matches your increment
- User confirmed server restart
- Fresh model loaded successfully
- Demo runs with new code behavior

---

**Remember**: The version number is the ONLY reliable way to confirm code changes are active!