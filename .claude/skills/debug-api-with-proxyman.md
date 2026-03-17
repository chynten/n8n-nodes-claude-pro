# Skill: Debug API Requests Using Proxyman HAR Capture

## When to use
Use this skill when the node's API calls are failing (especially HTTP 400/401/403) and the cause is unclear. This typically happens when Anthropic changes their API validation for OAuth setup-tokens.

## Prerequisites
Ask the user to:
1. Open **Proxyman** (or any HTTPS proxy) on their machine
2. Run a working **Claude Code CLI** command through the proxy:
   ```
   claude --proxy http://localhost:9090 "say hello"
   ```
3. In Proxyman, find the successful `POST https://api.anthropic.com/v1/messages` request
4. Export as **HAR file** (File → Export → HAR) and provide the file path

## Step-by-step analysis

### Step 1: Parse the HAR file
Extract all `POST /v1/messages` entries from the HAR:
```python
python3 -c "
import json
with open('<HAR_FILE>') as f:
    har = json.load(f)
for i, entry in enumerate(har['log']['entries']):
    req = entry['request']
    if '/v1/messages' in req['url'] and req['method'] == 'POST':
        print(f'Entry {i}: {req[\"url\"]} -> {entry[\"response\"][\"status\"]}')
"
```

### Step 2: Compare request headers
Extract headers from the working HAR request and compare against `buildAuthHeaders()` in `ClaudePro.node.ts`:

| Check | Where in code |
|-------|--------------|
| `authorization` format | `buildAuthHeaders()` |
| `anthropic-beta` list | `OAUTH_BETAS` constant |
| `user-agent` value | `CLAUDE_CLI_USER_AGENT` constant |
| `anthropic-version` | `buildAuthHeaders()` |
| `x-app` | `buildAuthHeaders()` |
| `anthropic-dangerous-direct-browser-access` | `buildAuthHeaders()` |
| URL query params (e.g. `?beta=true`) | `getMessagesUrl()` |

Check for any NEW headers in the HAR that our code doesn't send.

### Step 3: Compare request body
Extract the body from the working HAR and compare against `_buildRequestBody()`:

```python
python3 -c "
import json
with open('<HAR_FILE>') as f:
    har = json.load(f)
for entry in har['log']['entries']:
    req = entry['request']
    if '/v1/messages' in req['url'] and req['method'] == 'POST':
        body = json.loads(req['postData']['text'])
        # Show top-level keys and system prompt structure
        print('Top-level keys:', list(body.keys()))
        print('model:', body.get('model'))
        print('metadata:', json.dumps(body.get('metadata', 'NONE')))
        if 'system' in body:
            for i, s in enumerate(body['system'][:5]):
                t = s.get('text', '') if isinstance(s, dict) else str(s)
                print(f'system[{i}]: type={s.get(\"type\",\"?\")} len={len(t)} preview={repr(t[:150])}')
        if 'thinking' in body:
            print('thinking:', json.dumps(body['thinking']))
        break
"
```

Key body fields to check:
- `system` array — look for billing/internal system prompt entries
- `metadata` — check if `user_id` or other fields are present
- `thinking` — check format (`enabled` vs `adaptive`)
- `stream` — boolean
- Any new top-level keys we don't send

### Step 4: Isolate the difference
Create a minimal test script that sends requests WITH and WITHOUT each difference found. Test one variable at a time:

```javascript
// test-diff.mjs
const token = process.argv[2];
const tests = [
  { name: 'Without new field', body: { /* baseline */ } },
  { name: 'With new field', body: { /* baseline + new field */ } },
];
// Run each and check SUCCESS vs FAILED
```

### Step 5: Apply the fix
Once the required field is identified:
1. Update the relevant constant or function in `ClaudePro.node.ts`
2. If it's a header change → update `buildAuthHeaders()` and `credentials/ClaudeProApi.credentials.ts`
3. If it's a body change → update `_buildRequestBody()`
4. Bump version in `package.json`
5. `npm run build` → `npm publish`

## Known discoveries (update as new ones are found)

### v2.0.20: Billing system prompt required for Sonnet/Opus
- **Symptom:** HTTP 400 `invalid_request_error` with vague "Error" message for all models except Haiku
- **Root cause:** OAuth setup-tokens need `x-anthropic-billing-header` in the system prompt
- **Fix:** Prepend `x-anthropic-billing-header: cc_version=2.1.77; cc_entrypoint=cli;` as first system prompt entry
- **Required fields:** `cc_version` and `cc_entrypoint=cli` (without cc_entrypoint, API rejects as "reserved keyword")
- **Not required:** `metadata.user_id`, X-Stainless headers, `cch` hash

### v2.0.19: Headers and URL alignment with CLI
- **Fix:** Added `?beta=true` URL param, 8 beta headers, `user-agent`, `x-app`, switched from `https.request` to `fetch`
