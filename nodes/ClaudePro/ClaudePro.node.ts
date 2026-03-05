import {
  IExecuteFunctions,
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
  NodeOperationError,
} from 'n8n-workflow';

interface AnthropicContentBlock {
  type: string;
  text?: string;
}

interface AnthropicUsage {
  input_tokens: number;
  output_tokens: number;
}

interface AnthropicResponse {
  id: string;
  model: string;
  content: AnthropicContentBlock[];
  stop_reason: string;
  usage: AnthropicUsage;
}

interface SSEParsedResult {
  text: string;
  thinking: string;
  model: string;
  stopReason: string;
  usage: { inputTokens: number; outputTokens: number };
}

function isOAuthToken(token: string): boolean {
  return token.startsWith('sk-ant-oat');
}

function buildAuthHeaders(token: string): Record<string, string> {
  const headers: Record<string, string> = {
    'anthropic-version': '2023-06-01',
    'content-type': 'application/json',
  };
  if (isOAuthToken(token)) {
    headers['Authorization'] = `Bearer ${token}`;
    headers['anthropic-beta'] = 'oauth-2025-04-20,claude-code-20250219';
    headers['anthropic-dangerous-direct-browser-access'] = 'true';
  } else {
    headers['x-api-key'] = token;
  }
  return headers;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function formatApiError(error: any): string {
  const statusCode = error.httpCode || error.statusCode || error.code;

  if (statusCode === 401) {
    return 'Authentication failed. Your Claude Code token (sk-ant-oat01-*) may have expired. '
      + 'Run `claude setup-token` again to get a fresh token.';
  }
  if (statusCode === 403) {
    return 'Token does not have permission for this model.';
  }
  if (statusCode === 429) {
    return 'Rate limited by Anthropic API. Retry after a delay.';
  }
  if (statusCode === 529 || statusCode === 500) {
    return 'Anthropic API error. Retry later.';
  }
  if (statusCode === 400) {
    const msg = error.message || error.body?.error?.message || 'Bad request';
    return `Anthropic API error: ${msg}`;
  }

  return error.message || 'Unknown error occurred';
}

function parseSSE(raw: string, fallbackModel: string): SSEParsedResult {
  const lines = raw.split('\n');
  let text = '';
  let thinking = '';
  let model = fallbackModel;
  let stopReason = '';
  let inputTokens = 0;
  let outputTokens = 0;

  for (const line of lines) {
    if (!line.startsWith('data: ')) continue;

    const data = line.slice(6).trim();
    if (data === '[DONE]') break;

    let event;
    try {
      event = JSON.parse(data);
    } catch {
      continue;
    }

    switch (event.type) {
      case 'message_start':
        if (event.message?.model) model = event.message.model;
        if (event.message?.usage?.input_tokens) {
          inputTokens = event.message.usage.input_tokens;
        }
        break;

      case 'content_block_delta':
        if (event.delta?.type === 'text_delta') {
          text += event.delta.text || '';
        } else if (event.delta?.type === 'thinking_delta') {
          thinking += event.delta.thinking || '';
        }
        break;

      case 'message_delta':
        if (event.delta?.stop_reason) stopReason = event.delta.stop_reason;
        if (event.usage?.output_tokens) outputTokens = event.usage.output_tokens;
        break;
    }
  }

  return {
    text,
    thinking,
    model,
    stopReason,
    usage: { inputTokens, outputTokens },
  };
}

async function executeStreamingFallback(
  context: IExecuteFunctions,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  body: Record<string, any>,
): Promise<INodeExecutionData> {
  const https = require('https');

  const credentials = await context.getCredentials('claudeProApi');
  const token = credentials.setupToken as string;

  return new Promise((resolve, reject) => {
    const postData = JSON.stringify(body);

    const baseHeaders = buildAuthHeaders(token);
    const options = {
      hostname: 'api.anthropic.com',
      path: '/v1/messages',
      method: 'POST',
      headers: { ...baseHeaders, 'content-length': Buffer.byteLength(postData) },
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const req = https.request(options, (res: any) => {
      let rawData = '';

      res.on('data', (chunk: Buffer) => {
        rawData += chunk.toString();
      });

      res.on('end', () => {
        if (res.statusCode && res.statusCode >= 400) {
          const error = new Error(`HTTP ${res.statusCode}: ${rawData}`);
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (error as any).statusCode = res.statusCode;
          reject(error);
          return;
        }

        const parsed = parseSSE(rawData, body.model);
        resolve({ json: { ...parsed } });
      });
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    req.on('error', (error: any) => reject(error));
    req.write(postData);
    req.end();
  });
}

export class ClaudePro implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Claude Pro',
    name: 'claudePro',
    icon: 'file:claude-pro.svg',
    group: ['transform'],
    version: 1,
    subtitle: '={{$parameter["model"]}}',
    description: 'Send messages to Claude via setup-token authentication. No CLI required.',
    defaults: {
      name: 'Claude Pro',
    },
    inputs: ['main'],
    outputs: ['main'],
    credentials: [
      {
        name: 'claudeProApi',
        required: true,
      },
    ],
    properties: [
      {
        displayName: 'Model',
        name: 'model',
        type: 'options',
        options: [
          { name: 'Claude Opus 4', value: 'claude-opus-4-0-20250514' },
          { name: 'Claude Sonnet 4', value: 'claude-sonnet-4-20250514' },
          { name: 'Claude Haiku 3.5', value: 'claude-3-5-haiku-20241022' },
        ],
        default: 'claude-sonnet-4-20250514',
        description: 'The Claude model to use',
      },
      {
        displayName: 'Prompt',
        name: 'prompt',
        type: 'string',
        typeOptions: { rows: 6 },
        default: '',
        required: true,
        description: 'The message to send to Claude. Supports n8n expressions.',
      },
      {
        displayName: 'System Prompt',
        name: 'systemPrompt',
        type: 'string',
        typeOptions: { rows: 4 },
        default: '',
        description: 'Optional system prompt to set context for Claude',
      },
      {
        displayName: 'Max Tokens',
        name: 'maxTokens',
        type: 'number',
        typeOptions: { minValue: 1, maxValue: 128000 },
        default: 4096,
        description: 'Maximum number of tokens in the response',
      },
      {
        displayName: 'Temperature',
        name: 'temperature',
        type: 'number',
        typeOptions: { minValue: 0, maxValue: 2, numberPrecision: 1 },
        default: 1.0,
        description: 'Controls randomness. Lower values are more deterministic.',
      },
      {
        displayName: 'Streaming',
        name: 'streaming',
        type: 'boolean',
        default: false,
        description:
          'Whether to use SSE streaming. Response is still collected fully before output.',
      },
      {
        displayName: 'Extended Thinking',
        name: 'extendedThinking',
        type: 'boolean',
        default: false,
        description: 'Whether to enable chain-of-thought reasoning',
      },
      {
        displayName: 'Thinking Budget',
        name: 'thinkingBudget',
        type: 'number',
        typeOptions: { minValue: 1024, maxValue: 128000 },
        default: 10000,
        displayOptions: {
          show: {
            extendedThinking: [true],
          },
        },
        description: 'Max tokens for the thinking process',
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const returnData: INodeExecutionData[] = [];

    const credentials = await this.getCredentials('claudeProApi');
    const token = credentials.setupToken as string;
    const authHeaders = buildAuthHeaders(token);

    for (let i = 0; i < items.length; i++) {
      try {
        const model = this.getNodeParameter('model', i) as string;
        const prompt = this.getNodeParameter('prompt', i) as string;
        const systemPrompt = this.getNodeParameter('systemPrompt', i, '') as string;
        const maxTokens = this.getNodeParameter('maxTokens', i, 4096) as number;
        const temperature = this.getNodeParameter('temperature', i, 1.0) as number;
        const streaming = this.getNodeParameter('streaming', i, false) as boolean;
        const extendedThinking = this.getNodeParameter('extendedThinking', i, false) as boolean;
        const thinkingBudget = extendedThinking
          ? (this.getNodeParameter('thinkingBudget', i, 10000) as number)
          : undefined;

        if (!prompt.trim()) {
          throw new NodeOperationError(this.getNode(), 'Prompt cannot be empty', {
            itemIndex: i,
          });
        }

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const body: Record<string, any> = {
          model,
          max_tokens: maxTokens,
          messages: [{ role: 'user', content: prompt }],
          stream: streaming,
        };

        if (systemPrompt.trim()) {
          body.system = systemPrompt;
        }

        if (extendedThinking && thinkingBudget) {
          body.thinking = {
            type: 'enabled',
            budget_tokens: thinkingBudget,
          };
          // Temperature must be 1 when extended thinking is enabled
          body.temperature = 1;
        } else {
          body.temperature = temperature;
        }

        let result: INodeExecutionData;

        if (streaming) {
          try {
            const response = await this.helpers.httpRequest({
              method: 'POST',
              url: 'https://api.anthropic.com/v1/messages',
              body,
              returnFullResponse: true,
              encoding: 'text',
              json: false,
              headers: authHeaders,
            });

            const rawBody =
              typeof response.body === 'string' ? response.body : JSON.stringify(response.body);
            const parsed = parseSSE(rawBody, model);
            result = { json: { ...parsed } };
          } catch {
            // Fallback: use native https for SSE if n8n helpers don't handle it
            result = await executeStreamingFallback(this, body);
          }
        } else {
          const response = (await this.helpers.httpRequest({
            method: 'POST',
            url: 'https://api.anthropic.com/v1/messages',
            body,
            headers: authHeaders,
          })) as AnthropicResponse;

          const textBlocks = response.content.filter((b) => b.type === 'text');
          const thinkingBlocks = response.content.filter((b) => b.type === 'thinking');

          result = {
            json: {
              text: textBlocks.map((b) => b.text || '').join(''),
              thinking: thinkingBlocks.map((b) => b.text || '').join(''),
              model: response.model,
              stopReason: response.stop_reason,
              usage: {
                inputTokens: response.usage.input_tokens,
                outputTokens: response.usage.output_tokens,
              },
            },
          };
        }

        returnData.push(result);
      } catch (error) {
        if (this.continueOnFail()) {
          returnData.push({
            json: { error: formatApiError(error) },
            pairedItem: { item: i },
          });
          continue;
        }

        if (error instanceof NodeOperationError) {
          throw error;
        }
        throw new NodeOperationError(this.getNode(), formatApiError(error), { itemIndex: i });
      }
    }

    return [returnData];
  }
}
