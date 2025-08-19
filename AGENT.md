# AGENT.md

# AGENT INSTRUCTIONS

## 🚨 READ FIRST: AI Assistant Boundaries 🚨

**MANDATORY**: Before proceeding with ANY task, read `AI_ASSISTANT_BOUNDARIES.md`

This document prevents common AI assistant mistakes including:
- Process manipulation attempts
- Direct code execution bypassing MCP
- System administration overreach
- Testing without proper restart protocols

**You are a CODE ASSISTANT, not a SYSTEM ADMINISTRATOR**

## 🚨 MCP SERVER RESTART REQUIREMENT 🚨

**CRITICAL POLICY VIOLATION WARNING**: Modifying MCP server code and immediately calling MCP tools is **STRICTLY FORBIDDEN** and violates the code modification policy.

**MANDATORY PROTOCOL**: Every time you modify ANY MCP server code, you MUST:
1. **STOP IMMEDIATELY** - Do not call any MCP tools
2. Increment the version number in the version tool
3. Ask the user to restart the MCP server in their IDE
4. **WAIT for user confirmation** of restart
5. Verify restart by calling `version` tool first
6. Restart the entire demo workflow from the beginning

**⚠️ ABSOLUTE PROHIBITION**: Never test code changes without restart verification!
**⚠️ RESOURCE WASTE**: Testing without restart wastes computational resources on old code!
**⚠️ FALSE DEBUGGING**: You will debug non-existent problems in unchanged code!

## Automation & Prevention System

🤖 **FOR AI ASSISTANTS**: See `mcp-server/MCP_RESTART_AUTOMATION.md` for:
- Automated reminder triggers
- Mandatory verification protocol
- Prevention of testing without restart
- User instruction templates

**ALWAYS reference the automation guide before MCP testing!**

See `mcp-server/MCP_RESTART_CHECKLIST.md` for detailed protocol.

## On Testing ##

Something untested is never successful. You cannot declare success without testing it first.

## Workflow: Never Fake Anything. Otherwise, Proceed Without Asking.

Never fake anything. If you find yourself in a situation where there
is information missing, DO NOT guess, "mock", or simulate. Instead,
STOP and ask the user for clarification. Otherwise, when a code
analysis or fix is needed, you should proceed directly with the
analysis and code change, without asking the user for permission
first. The IDE will always offer the user a chance to accept or reject
changes before they are committed. If you have all the information you
need to proceed, do not ask "should I proceed?"... just do the work.

## Creating and Modifying Specifications

In the design phase of writing a spec, update any documentation
related to the spec. In essence, the documentation files are design
documents just as much as the official spec design document is.

## Documenting Development Progress

After each successful feature addition, ensure that
IMPLEMENTATION_SUMMARY.md is up to date.

## Using the shell

The shell is 'tcsh', not 'bash'. Always wait for commands to
finish. You are not allowed to run any code from the shell. The only
way you are allowed to run code is via the MCP server tools.

## 🚨 CRITICAL: AI Assistant Constraints 🚨

**NEVER DO THESE ACTIONS** - Common AI Assistant Mistakes:

❌ **Process Manipulation**: NEVER attempt to:
- Kill processes with `kill`, `pkill`, or `killall`
- Check running processes with `ps aux | grep`
- Start/stop/restart services directly
- Read PID files to manipulate processes
- Use `systemctl`, `service`, or similar commands

❌ **Direct Code Execution**: NEVER attempt to:
- Run Python scripts directly from shell
- Execute any code outside MCP tools
- Test implementations without MCP server restart

❌ **File System Operations**: NEVER attempt to:
- Manipulate files outside the provided file tools
- Use shell commands for file operations

✅ **ONLY ALLOWED INTERACTIONS**:
- Use MCP server tools for ALL code interaction
- Use provided file editing tools for code changes
- Follow MCP restart protocol for testing
- Ask user to manually restart services

**Remember**: You are a code assistant, NOT a system administrator. Your role is limited to code analysis and modification through the provided tools.

## Using TypeScript

Never use TypeScript; use vanilla JavaScript.

## Using Smalltalk

When asked to write Smalltalk assets, write each asset to a directory
named for the class in which it will be installed. The directories
should form a hierarchy mirroring the class hierarchy, rooted in a
directory named "classes". Write each class comment in a file named
"COMMENT.md". Write each method in a file named for the selector that
method will use. The content of a method file should use Smalltalk
syntax. When writing a new method, writing a comment but no code is
acceptable. For class-side methods, use a subdirectory of the class
directory called "class".

## Using JavaScript

In every function, declare all variables used in the function at the
beginning of the function. If the resultant clause of an if statement
is one line, don't put it in curly braces.

Never bundle sources. Never use webpack or anything like it.

## Using Cosmos Graph (@cosmos.gl/graph)

The API documentation is at /Users/craig/forks/graph/src/stories/. Use
@interacta/css-labels for labels.

## Using the mcp-server MCP Server

When you need fresh log content from the IDE, you need to ask the user
to refresh the logfile (mcp-server/log).

## When generating HTML

Don't declare success until the user has given feedback on the
result. There's an MCP tool for opening a browser. HTML files are
generated by the MCP server tools. You shouldn't be editing HTML files
directly.

## WASM Tools

The only WASM tools you may use are 'wasm-tools' and 'wasm-opt'. You
may not use wat2wasm or any other WASM tools.

---

