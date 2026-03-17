import {
  IAuthenticateGeneric,
  ICredentialTestRequest,
  ICredentialType,
  INodeProperties,
} from 'n8n-workflow';

export class ClaudeProApi implements ICredentialType {
  name = 'claudeProApi';
  displayName = 'Claude Pro Setup Token';
  documentationUrl = 'https://docs.anthropic.com/en/docs/claude-code/cli-reference#claude-setup-token';

  properties: INodeProperties[] = [
    {
      displayName: 'Setup Token',
      name: 'setupToken',
      type: 'string',
      typeOptions: { password: true },
      default: '',
      required: true,
      description:
        'OAuth token from "claude setup-token" (sk-ant-oat01-*). Tokens expire every few hours — regenerate with `claude setup-token`.',
    },
  ];

  authenticate: IAuthenticateGeneric = {
    type: 'generic',
    properties: {
      headers: {
        'Authorization': '=Bearer {{$credentials.setupToken}}',
        'anthropic-beta': 'claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,redact-thinking-2026-02-12,context-management-2025-06-27,prompt-caching-scope-2026-01-05,advanced-tool-use-2025-11-20,effort-2025-11-24',
        'anthropic-dangerous-direct-browser-access': 'true',
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
        'accept': 'application/json',
        'user-agent': 'claude-cli/2.1.77 (external, cli)',
        'x-app': 'cli',
      },
    },
  };

  test: ICredentialTestRequest = {
    request: {
      baseURL: 'https://api.anthropic.com',
      url: '/v1/models',
      method: 'GET',
    },
  };
}
