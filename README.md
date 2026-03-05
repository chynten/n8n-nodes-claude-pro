# n8n-nodes-claude-pro

An n8n community node for calling Claude (Anthropic) using a **setup-token** for authentication. No Claude CLI installation required on the n8n server.

## Why this node?

Existing n8n community nodes for Claude require the Claude Code CLI installed on the server and use browser-based OAuth. This doesn't work for hosted/remote n8n instances where you can't run `claude login` interactively.

This node uses a different approach: generate a setup-token on your local machine, paste it into n8n credentials, and the node calls the Claude API directly with Bearer auth.

## Prerequisites

1. Install the [Claude CLI](https://docs.anthropic.com/en/docs/claude-code/overview) on your **local machine**
2. Log in: `claude login`
3. Generate a setup token: `claude setup-token`
4. Copy the token output

## Installation

### Community Nodes (recommended)

1. Go to **Settings > Community Nodes** in your n8n instance
2. Search for `n8n-nodes-claude-pro`
3. Install

### Manual Installation

```bash
cd ~/.n8n/custom
npm install n8n-nodes-claude-pro
# or link a local clone:
ln -s /path/to/claude-pro-node ~/.n8n/custom/node_modules/n8n-nodes-claude-pro
```

Restart n8n after installation.

## Configuration

### Credentials

1. In n8n, go to **Credentials > New Credential**
2. Search for "Claude Pro Setup Token"
3. Paste your setup token from `claude setup-token`
4. Click **Test** to verify the token works

### Node Parameters

| Parameter | Default | Description |
|---|---|---|
| Model | Claude Sonnet 4 | Choose between Opus 4, Sonnet 4, Haiku 3.5 |
| Prompt | — | The message to send (required). Supports expressions. |
| System Prompt | empty | Optional system-level instructions |
| Max Tokens | 4096 | Maximum response length (1–128,000) |
| Temperature | 1.0 | Randomness control (0.0–2.0) |
| Streaming | false | Use SSE streaming (collected before output) |
| Extended Thinking | false | Enable chain-of-thought reasoning |
| Thinking Budget | 10000 | Token budget for thinking (shown when thinking enabled) |

### Output

Each item returns:

```json
{
  "text": "Claude's response",
  "thinking": "Chain-of-thought (if extended thinking enabled)",
  "model": "claude-sonnet-4-20250514",
  "stopReason": "end_turn",
  "usage": {
    "inputTokens": 25,
    "outputTokens": 150
  }
}
```

## Error Handling

The node provides clear error messages:

- **401**: Setup token is invalid or expired. Run `claude setup-token` again.
- **403**: Token does not have permission for the selected model.
- **429**: Rate limited. Retry after a delay.
- **400**: Forwards the Anthropic error message.
- **500/529**: Anthropic API error. Retry later.

Enable **Continue On Fail** in the node settings to receive errors as JSON output instead of stopping the workflow.

## Development

```bash
git clone <repo-url>
cd claude-pro-node
npm install
npm run build
```

To test locally with n8n:

```bash
ln -s $(pwd) ~/.n8n/custom/node_modules/n8n-nodes-claude-pro
# restart n8n
```

## License

MIT
