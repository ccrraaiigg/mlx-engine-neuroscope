# AGENT.md

# AGENT INSTRUCTIONS

## Workflow: Never Fake Anything. Otherwise, Proceed Without Asking.

Never fake anything. If you find yourself in a situation where there
is information missing, DO NOT guess, "mock", or simulate. Instead,
STOP and ask the user for clarification. Otherwise, when a code
analysis or fix is needed, you should proceed directly with the
analysis and code change, without asking the user for permission
first. Cursor will always offer the user a chance to accept or reject
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

The shell is 'tcsh', not 'bash'. Always wait for commands to finish.

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

If the resultant clause of an if statement is one line, don't put it
in curly braces.

Never bundle sources. Never use webpack or anything like it.

## Using @cosmos.gl/graph

The API documentation is at /Users/craig/forks/graph/src/stories/.

## WASM Tools

The only WASM tools you may use are 'wasm-tools' and 'wasm-opt'. You
may not use wat2wasm or any other WASM tools.

---

