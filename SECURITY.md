# Security Policy

## Supported Versions

| Version | Supported          |
|---------|-------------------|
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

Only the latest 1.0.x release is actively supported with security updates.

## Reporting a Vulnerability

If you discover a security vulnerability, please email **sameerpashasyed17@gmail.com** with:
- Description of the vulnerability
- Steps to reproduce (if applicable)
- Potential impact
- Suggested fix (if you have one)

We commit to responding to security reports within 7 days and will work with you to resolve the issue responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

## Security Scope

### What We Consider a Security Issue
- Exposure of prompt data or user input through logs, files, or unintended channels
- SQLite injection vulnerabilities that could compromise local data
- SSRF (Server-Side Request Forgery) vulnerabilities in webhook handlers
- Authentication or authorization flaws

### What We Do NOT Consider a Security Issue
- Performance issues or optimization suggestions
- Missing features or enhancement requests
- Configuration misunderstandings

## Data Security

ConsistencyGuard stores all prompts locally in SQLite and never transmits them anywhere except to your configured LLM provider. Your data remains under your control at all times.
