import {
  ILoadOptionsFunctions,
  INodeExecutionData,
  INodePropertyOptions,
  INodeType,
  INodeTypeDescription,
  ISupplyDataFunctions,
  NodeConnectionType,
  NodeConnectionTypes,
  SupplyData,
} from 'n8n-workflow';

import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { BaseChatModelCallOptions, BindToolsInput } from '@langchain/core/language_models/chat_models';
import type { BaseMessage } from '@langchain/core/messages';
import { AIMessageChunk } from '@langchain/core/messages';
import type { ChatResult } from '@langchain/core/outputs';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseLanguageModelInput } from '@langchain/core/language_models/base';
import { convertToOpenAITool } from '@langchain/core/utils/function_calling';
import type { Runnable } from '@langchain/core/runnables';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnthropicTool = { name: string; description: string; input_schema: Record<string, any> };

interface AnthropicContentBlock {
  type: string;
  text?: string;
  id?: string;
  name?: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  input?: Record<string, any>;
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

const MODEL_MAX_TOKENS: Record<string, number> = {
  // Claude 4.6 family
  'claude-opus-4-6': 128000,
  'claude-sonnet-4-6': 64000,
  // Claude 4.5 family
  'claude-opus-4-5-20251101': 64000,
  'claude-opus-4-5': 64000,
  'claude-sonnet-4-5-20250929': 64000,
  'claude-sonnet-4-5': 64000,
  'claude-haiku-4-5-20251001': 64000,
  'claude-haiku-4-5': 64000,
  // Claude 4.1 family
  'claude-opus-4-1-20250805': 32000,
  'claude-opus-4-1': 32000,
  // Claude 4.0 family
  'claude-opus-4-20250514': 32000,
  'claude-opus-4-0': 32000,
  'claude-sonnet-4-20250514': 64000,
  'claude-sonnet-4-0': 64000,
  // Claude 3.7 family
  'claude-3-7-sonnet-20250219': 16384,
  'claude-3-7-sonnet-latest': 16384,
  // Claude 3.5 family
  'claude-3-5-sonnet-20241022': 8192,
  'claude-3-5-sonnet-20240620': 8192,
  'claude-3-5-sonnet-latest': 8192,
  'claude-3-5-haiku-20241022': 8192,
  'claude-3-5-haiku-latest': 8192,
  // Claude 3 family
  'claude-3-opus-20240229': 4096,
  'claude-3-opus-latest': 4096,
  'claude-3-sonnet-20240229': 4096,
  'claude-3-haiku-20240307': 4096,
};

const DEFAULT_MAX_TOKENS = 16384;
const CLAUDE_CLI_USER_AGENT = 'claude-cli/2.1.77 (external, cli)';
const OAUTH_BETAS = 'claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,redact-thinking-2026-02-12,context-management-2025-06-27,prompt-caching-scope-2026-01-05,advanced-tool-use-2025-11-20,effort-2025-11-24';
const OAUTH_BILLING_HEADER = 'x-anthropic-billing-header: cc_version=2.1.77; cc_entrypoint=cli;';

function getMaxTokensForModel(modelId: string): number {
  if (MODEL_MAX_TOKENS[modelId]) {
    return MODEL_MAX_TOKENS[modelId];
  }
  // Pattern-based fallback for new model versions
  if (modelId.includes('opus-4-6') || modelId.includes('opus-4-5')) return 64000;
  if (modelId.includes('sonnet-4-6') || modelId.includes('sonnet-4-5')) return 64000;
  if (modelId.includes('haiku-4-5') || modelId.includes('haiku-4')) return 64000;
  if (modelId.includes('opus-4-1')) return 32000;
  if (modelId.includes('opus-4')) return 32000;
  if (modelId.includes('sonnet-4')) return 64000;
  if (modelId.startsWith('claude-3-7')) return 16384;
  if (modelId.startsWith('claude-3-5')) return 8192;
  if (modelId.startsWith('claude-3-')) return 4096;
  return DEFAULT_MAX_TOKENS;
}

function isOAuthToken(token: string): boolean {
  return token.startsWith('sk-ant-oat');
}

function buildAuthHeaders(token: string): Record<string, string> {
  const headers: Record<string, string> = {
    'accept': 'application/json',
    'anthropic-version': '2023-06-01',
    'content-type': 'application/json',
  };

  if (isOAuthToken(token)) {
    headers['authorization'] = `Bearer ${token}`;
    headers['anthropic-beta'] = OAUTH_BETAS;
    headers['anthropic-dangerous-direct-browser-access'] = 'true';
    headers['user-agent'] = CLAUDE_CLI_USER_AGENT;
    headers['x-app'] = 'cli';
  } else {
    headers['x-api-key'] = token;
    headers['anthropic-beta'] = 'interleaved-thinking-2025-05-14';
  }
  return headers;
}

function getMessagesUrl(token: string): string {
  return isOAuthToken(token)
    ? 'https://api.anthropic.com/v1/messages?beta=true'
    : 'https://api.anthropic.com/v1/messages';
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function convertMessages(messages: BaseMessage[]): Array<{ role: string; content: any }> {
  return messages.map((msg) => {
    const msgType = msg._getType();

    if (msgType === 'tool') {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const toolMsg = msg as any;
      return {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: toolMsg.tool_call_id || toolMsg.additional_kwargs?.tool_call_id,
            content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content),
          },
        ],
      };
    }

    let role: string;
    switch (msgType) {
      case 'human':
        role = 'user';
        break;
      case 'ai':
        role = 'assistant';
        break;
      default:
        role = 'user';
    }

    // For AI messages with tool calls, reconstruct Anthropic content blocks
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const aiMsg = msg as any;
    if (msgType === 'ai' && aiMsg.tool_calls && aiMsg.tool_calls.length > 0) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const content: any[] = [];
      const textContent = typeof msg.content === 'string' ? msg.content : '';
      if (textContent) {
        content.push({ type: 'text', text: textContent });
      }
      for (const tc of aiMsg.tool_calls) {
        content.push({
          type: 'tool_use',
          id: tc.id,
          name: tc.name,
          input: tc.args,
        });
      }
      return { role, content };
    }

    const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
    return { role, content };
  }).filter((msg) => {
    // Filter out messages with empty content (strings only; arrays like tool_result are kept)
    if (typeof msg.content === 'string' && msg.content.trim() === '') return false;
    return true;
  });
}

function extractSystemMessage(messages: BaseMessage[]): { system?: string; filtered: BaseMessage[] } {
  const systemMsgs = messages.filter((m) => m._getType() === 'system');
  const filtered = messages.filter((m) => m._getType() !== 'system');
  const system = systemMsgs.length > 0
    ? systemMsgs.map((m) => typeof m.content === 'string' ? m.content : JSON.stringify(m.content)).join('\n')
    : undefined;
  return { system, filtered };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function* parseSSEEvents(stream: any): AsyncGenerator<{ event: string; data: any }> {
  let buffer = '';
  for await (const chunk of stream) {
    buffer += chunk.toString();
    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';
    for (const part of parts) {
      if (!part.trim()) continue;
      let event = '';
      let data = '';
      for (const line of part.split('\n')) {
        if (line.startsWith('event:')) {
          event = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          data = line.slice(5).trim();
        }
      }
      if (event && data) {
        try {
          yield { event, data: JSON.parse(data) };
        } catch {
          // skip malformed JSON
        }
      }
    }
  }
}

function openAiToolToAnthropic(tools: BindToolsInput[]): AnthropicTool[] {
  return tools.map((tool) => {
    const openAiTool = convertToOpenAITool(tool);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const params: Record<string, any> = { ...(openAiTool.function.parameters || { type: 'object', properties: {} }) };
    // Anthropic API rejects $schema in tool input_schema
    delete params.$schema;
    return {
      name: openAiTool.function.name,
      description: openAiTool.function.description || '',
      input_schema: params,
    };
  });
}

interface ClaudeProCallOptions extends BaseChatModelCallOptions {
  tools?: AnthropicTool[];
}

interface N8nExecutionContext {
  addInputData: ISupplyDataFunctions['addInputData'];
  addOutputData: ISupplyDataFunctions['addOutputData'];
  connectionType: NodeConnectionType;
}

interface ClaudeProModelInput {
  token: string;
  modelId: string;
  maxTokens: number;
  temperature: number;
  topP?: number;
  topK?: number;
  extendedThinking: boolean;
  thinkingBudget?: number;
  timeout: number;
  maxRetries: number;
  tools?: AnthropicTool[];
  n8nContext?: N8nExecutionContext;
}

class ClaudeProChatModel extends BaseChatModel<ClaudeProCallOptions> {
  private token: string;
  private modelId: string;
  private maxTokens: number;
  private temperature: number;
  private topP?: number;
  private topK?: number;
  private extendedThinking: boolean;
  private thinkingBudget?: number;
  private timeout: number;
  private maxRetries: number;
  private boundTools?: AnthropicTool[];
  private n8nContext?: N8nExecutionContext;

  lc_serializable = false;

  constructor(fields: ClaudeProModelInput) {
    super({});
    this.token = fields.token;
    this.modelId = fields.modelId;
    this.maxTokens = fields.maxTokens;
    this.temperature = fields.temperature;
    this.topP = fields.topP;
    this.topK = fields.topK;
    this.extendedThinking = fields.extendedThinking;
    this.thinkingBudget = fields.thinkingBudget;
    this.timeout = fields.timeout;
    this.maxRetries = fields.maxRetries;
    this.boundTools = fields.tools;
    this.n8nContext = fields.n8nContext;
  }

  _llmType(): string {
    return 'claude-pro';
  }

  bindTools(
    tools: BindToolsInput[],
    _kwargs?: Partial<ClaudeProCallOptions>,
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, ClaudeProCallOptions> {
    const anthropicTools = openAiToolToAnthropic(tools);
    return new ClaudeProChatModel({
      token: this.token,
      modelId: this.modelId,
      maxTokens: this.maxTokens,
      temperature: this.temperature,
      topP: this.topP,
      topK: this.topK,
      extendedThinking: this.extendedThinking,
      thinkingBudget: this.thinkingBudget,
      timeout: this.timeout,
      maxRetries: this.maxRetries,
      tools: anthropicTools,
      n8nContext: this.n8nContext,
    }) as unknown as Runnable<BaseLanguageModelInput, AIMessageChunk, ClaudeProCallOptions>;
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    let runIndex: number | undefined;
    if (this.n8nContext) {
      const inputPayload: INodeExecutionData[] = messages.map((msg) => ({
        json: { role: msg._getType(), content: msg.content },
      }));
      const result = this.n8nContext.addInputData(this.n8nContext.connectionType, [inputPayload]);
      runIndex = result.index;
    }

    try {
      const chatResult = await this._withRetry(() => this._callApi(messages, options));

      if (this.n8nContext && runIndex !== undefined) {
        const outputPayload: INodeExecutionData[] = chatResult.generations.map((g) => ({
          json: {
            text: g.text,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            toolCalls: (g.message as any).tool_calls || [],
            tokenUsage: g.message.additional_kwargs?.usage || {},
          },
        }));
        this.n8nContext.addOutputData(this.n8nContext.connectionType, runIndex, [outputPayload]);
      }

      return chatResult;
    } catch (error) {
      if (this.n8nContext && runIndex !== undefined) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        this.n8nContext.addOutputData(this.n8nContext.connectionType, runIndex, error as any);
      }
      throw error;
    }
  }

  private _isRetryableError(error: unknown): boolean {
    if (error instanceof Error) {
      const msg = error.message;
      if (msg.includes('HTTP 5') || msg.includes('timed out')) return true;
      if (msg.includes('ECONNRESET') || msg.includes('ETIMEDOUT') || msg.includes('ECONNREFUSED')) return true;
    }
    return false;
  }

  private async _withRetry<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: unknown;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        if (attempt < this.maxRetries && this._isRetryableError(error)) {
          await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
          continue;
        }
        throw error;
      }
    }
    throw lastError;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _buildRequestBody(messages: BaseMessage[], options: this['ParsedCallOptions']): Record<string, any> {
    const { system, filtered } = extractSystemMessage(messages);
    const apiMessages = convertMessages(filtered);

    // Claude Opus 4.6 does not support prefilling assistant messages
    if (this.modelId.includes('opus-4-6') && apiMessages.length > 0) {
      if (apiMessages[apiMessages.length - 1].role === 'assistant') {
        apiMessages.pop();
      }
    }

    // Ensure messages array is non-empty
    if (apiMessages.length === 0) {
      apiMessages.push({ role: 'user', content: 'Hello' });
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const body: Record<string, any> = {
      model: this.modelId,
      max_tokens: this.maxTokens,
      messages: apiMessages,
    };

    // OAuth tokens require a billing header in the system prompt for Sonnet/Opus access
    if (isOAuthToken(this.token)) {
      const systemParts: Array<{ type: string; text: string }> = [
        { type: 'text', text: OAUTH_BILLING_HEADER },
      ];
      if (system) {
        systemParts.push({ type: 'text', text: system });
      }
      body.system = systemParts;
    } else if (system) {
      body.system = system;
    }

    const tools = (options.tools && options.tools.length > 0) ? options.tools : this.boundTools;
    if (tools && tools.length > 0) {
      // Strip $schema from all tool input_schema — Anthropic API rejects it
      body.tools = tools.map((t) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const schema: Record<string, any> = { ...t.input_schema };
        delete schema.$schema;
        return { ...t, input_schema: schema };
      });
    }

    if (this.extendedThinking && this.thinkingBudget) {
      // budget_tokens must be strictly less than max_tokens and at least 1024 per Anthropic API
      const effectiveBudget = Math.max(1024, Math.min(this.thinkingBudget, this.maxTokens - 1));
      body.thinking = { type: 'enabled', budget_tokens: effectiveBudget };
      body.temperature = 1;
    } else {
      body.temperature = this.temperature;
      if (this.topP !== undefined && this.topP < 1) {
        body.top_p = this.topP;
      }
      if (this.topK !== undefined && this.topK > -1) {
        body.top_k = this.topK;
      }
    }

    return body;
  }

  private async _callApi(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
  ): Promise<ChatResult> {
    const body = this._buildRequestBody(messages, options);
    const headers = buildAuthHeaders(this.token);
    const postData = JSON.stringify(body);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    let response: AnthropicResponse;
    try {
      const res = await fetch(getMessagesUrl(this.token), {
        method: 'POST',
        headers,
        body: postData,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!res.ok) {
        const rawData = await res.text();
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const debugHeaders: Record<string, any> = { ...headers };
        if (debugHeaders['x-api-key']) debugHeaders['x-api-key'] = '***masked***';
        if (debugHeaders['authorization']) debugHeaders['authorization'] = '***masked***';
        const debugInfo = `\n[Request Headers]: ${JSON.stringify(debugHeaders)}\n[Request Body]: ${JSON.stringify(body, null, 2)}`;
        try {
          const errorBody = JSON.parse(rawData);
          const errorType = errorBody?.error?.type || 'unknown';
          const errorMsg = errorBody?.error?.message || rawData;
          throw new Error(`Anthropic API error (HTTP ${res.status}) [${errorType}]: ${errorMsg}\n[Full API Response]: ${rawData}${debugInfo}`);
        } catch (parseError) {
          if (parseError instanceof Error && parseError.message.startsWith('Anthropic API error')) throw parseError;
          throw new Error(`Anthropic API error (HTTP ${res.status}): ${rawData}${debugInfo}`);
        }
      }
      response = await res.json() as AnthropicResponse;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Anthropic API request timed out after ${this.timeout}ms`);
      }
      throw error;
    }

    const textBlocks = response.content.filter((b) => b.type === 'text');
    const toolUseBlocks = response.content.filter((b) => b.type === 'tool_use');
    const text = textBlocks.map((b) => b.text || '').join('');

    const toolCalls: ToolCall[] = toolUseBlocks.map((b) => ({
      name: b.name || '',
      args: b.input || {},
      id: b.id || '',
      type: 'tool_call' as const,
    }));

    const aiMessage = new AIMessageChunk({
      content: text,
      tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
      additional_kwargs: {
        usage: {
          input_tokens: response.usage.input_tokens,
          output_tokens: response.usage.output_tokens,
        },
      },
    });

    return {
      generations: [
        {
          text,
          message: aiMessage,
        },
      ],
    };
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    let runIndex: number | undefined;
    if (this.n8nContext) {
      const inputPayload: INodeExecutionData[] = messages.map((msg) => ({
        json: { role: msg._getType(), content: msg.content },
      }));
      const result = this.n8nContext.addInputData(this.n8nContext.connectionType, [inputPayload]);
      runIndex = result.index;
    }

    try {
      const body = this._buildRequestBody(messages, options);
      body.stream = true;

      const headers = buildAuthHeaders(this.token);
      const postData = JSON.stringify(body);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      let res: Response;
      try {
        res = await fetch(getMessagesUrl(this.token), {
          method: 'POST',
          headers,
          body: postData,
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
      } catch (fetchError) {
        clearTimeout(timeoutId);
        if (fetchError instanceof Error && fetchError.name === 'AbortError') {
          throw new Error(`Anthropic API request timed out after ${this.timeout}ms`);
        }
        throw fetchError;
      }

      if (!res.ok) {
        const rawData = await res.text();
        const debugHeaders = { ...headers };
        if (debugHeaders['x-api-key']) debugHeaders['x-api-key'] = '***masked***';
        if (debugHeaders['authorization']) debugHeaders['authorization'] = '***masked***';
        const debugInfo = `\n[Request Headers]: ${JSON.stringify(debugHeaders)}\n[Request Body]: ${JSON.stringify(body, null, 2)}`;
        try {
          const errorBody = JSON.parse(rawData);
          const errorType = errorBody?.error?.type || 'unknown';
          const errorMsg = errorBody?.error?.message || rawData;
          throw new Error(`Anthropic API error (HTTP ${res.status}) [${errorType}]: ${errorMsg}\n[Full API Response]: ${rawData}${debugInfo}`);
        } catch (parseError) {
          if (parseError instanceof Error && parseError.message.startsWith('Anthropic API error')) throw parseError;
          throw new Error(`Anthropic API error (HTTP ${res.status}): ${rawData}${debugInfo}`);
        }
      }

      // Adapt fetch ReadableStream for SSE parsing
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const streamIterable: any = {
        [Symbol.asyncIterator]() {
          const reader = res.body!.getReader();
          return {
            async next() {
              const { done, value } = await reader.read();
              if (done) return { done: true, value: undefined };
              return { done: false, value: Buffer.from(value) };
            },
          };
        },
      };

      const toolBlocks = new Map<number, { id: string; name: string; partialJson: string }>();
      let fullText = '';

      for await (const { event, data } of parseSSEEvents(streamIterable)) {
        if (event === 'message_stop') break;

        if (event === 'error') {
          throw new Error(`Anthropic stream error: ${JSON.stringify(data)}`);
        }

        if (event === 'content_block_start') {
          if (data.content_block?.type === 'tool_use') {
            toolBlocks.set(data.index, {
              id: data.content_block.id,
              name: data.content_block.name,
              partialJson: '',
            });
          }
          continue;
        }

        if (event === 'content_block_delta') {
          if (data.delta?.type === 'text_delta') {
            const text = data.delta.text || '';
            fullText += text;
            const chunk = new ChatGenerationChunk({
              text,
              message: new AIMessageChunk({ content: text }),
            });
            yield chunk;
            await runManager?.handleLLMNewToken(text);
          } else if (data.delta?.type === 'input_json_delta') {
            const block = toolBlocks.get(data.index);
            if (block) {
              block.partialJson += data.delta.partial_json || '';
            }
          }
          continue;
        }

        if (event === 'content_block_stop') {
          const block = toolBlocks.get(data.index);
          if (block) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            let args: Record<string, any> = {};
            try {
              args = JSON.parse(block.partialJson);
            } catch {
              // use empty args if JSON is malformed
            }
            const toolCall: ToolCall = {
              name: block.name,
              args,
              id: block.id,
              type: 'tool_call' as const,
            };
            const chunk = new ChatGenerationChunk({
              text: '',
              message: new AIMessageChunk({ content: '', tool_calls: [toolCall] }),
            });
            yield chunk;
            toolBlocks.delete(data.index);
          }
          continue;
        }
      }

      if (this.n8nContext && runIndex !== undefined) {
        const outputPayload: INodeExecutionData[] = [{ json: { text: fullText } }];
        this.n8nContext.addOutputData(this.n8nContext.connectionType, runIndex, [outputPayload]);
      }
    } catch (error) {
      if (this.n8nContext && runIndex !== undefined) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        this.n8nContext.addOutputData(this.n8nContext.connectionType, runIndex, error as any);
      }
      throw error;
    }
  }
}

export class ClaudePro implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Claude Pro LM',
    name: 'claudePro',
    icon: 'file:claude-pro.svg',
    group: ['transform'],
    version: 2,
    subtitle: '={{$parameter["model"]}}',
    description: 'Language Model node for Claude via setup-token authentication.',
    defaults: {
      name: 'Claude Pro LM',
    },
    codex: {
      categories: ['AI'],
      subcategories: {
        AI: ['Language Models', 'Root Nodes'],
      },
    },
    inputs: [],
    outputs: [NodeConnectionTypes.AiLanguageModel],
    outputNames: ['Model'],
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
        typeOptions: {
          loadOptionsMethod: 'getModels',
        },
        default: '',
        description: 'The Claude model to use. List is fetched live from the API.',
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
      {
        displayName: 'Options',
        name: 'options',
        type: 'collection',
        placeholder: 'Add Option',
        default: {},
        options: [
          {
            displayName: 'Temperature',
            name: 'temperature',
            type: 'number',
            typeOptions: { minValue: 0, maxValue: 2, numberPrecision: 1 },
            default: 1.0,
            description: 'Controls randomness. Lower values are more deterministic. Ignored when Extended Thinking is enabled.',
          },
          {
            displayName: 'Top P',
            name: 'topP',
            type: 'number',
            typeOptions: { minValue: 0, maxValue: 1, numberPrecision: 1 },
            default: 1,
            description: 'Nucleus sampling: only consider tokens with cumulative probability up to this value. Ignored when Extended Thinking is enabled.',
          },
          {
            displayName: 'Top K',
            name: 'topK',
            type: 'number',
            typeOptions: { minValue: -1 },
            default: -1,
            description: 'Only sample from the top K most likely tokens. Set to -1 to disable. Ignored when Extended Thinking is enabled.',
          },
          {
            displayName: 'Max Tokens Override',
            name: 'maxTokensOverride',
            type: 'number',
            typeOptions: { minValue: -1 },
            default: -1,
            description: 'Override the automatic max output tokens for the selected model. Set to -1 to use the model default.',
          },
          {
            displayName: 'Timeout (ms)',
            name: 'timeout',
            type: 'number',
            typeOptions: { minValue: 1000 },
            default: 120000,
            description: 'Maximum time in milliseconds to wait for an API response',
          },
          {
            displayName: 'Max Retries',
            name: 'maxRetries',
            type: 'number',
            typeOptions: { minValue: 0, maxValue: 5 },
            default: 2,
            description: 'Number of retries on transient failures (5xx errors, network errors)',
          },
        ],
      },
    ],
  };

  methods = {
    loadOptions: {
      async getModels(this: ILoadOptionsFunctions): Promise<INodePropertyOptions[]> {
        const credentials = await this.getCredentials('claudeProApi');
        const token = credentials.setupToken as string;
        const headers = buildAuthHeaders(token);

        const response = await this.helpers.httpRequest({
          method: 'GET',
          url: 'https://api.anthropic.com/v1/models',
          headers,
        });

        const models = (response.data || []) as Array<{ id: string; display_name: string }>;
        return models.map((m) => {
          const maxTokens = getMaxTokensForModel(m.id);
          return {
            name: `${m.display_name} (max output: ${maxTokens.toLocaleString()})`,
            value: m.id,
          };
        });
      },
    },
  };

  async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
    const credentials = await this.getCredentials('claudeProApi');
    const token = credentials.setupToken as string;

    const modelId = this.getNodeParameter('model', itemIndex) as string;
    const extendedThinking = this.getNodeParameter('extendedThinking', itemIndex, false) as boolean;
    const thinkingBudget = extendedThinking
      ? (this.getNodeParameter('thinkingBudget', itemIndex, 10000) as number)
      : undefined;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const options = this.getNodeParameter('options', itemIndex, {}) as Record<string, any>;
    const temperature = (options.temperature ?? 1.0) as number;
    const topP = (options.topP ?? 1) as number;
    const topK = (options.topK ?? -1) as number;
    const maxTokensOverride = (options.maxTokensOverride ?? -1) as number;
    const timeout = (options.timeout ?? 120000) as number;
    const maxRetries = (options.maxRetries ?? 2) as number;
    const maxTokens = maxTokensOverride > 0 ? maxTokensOverride : getMaxTokensForModel(modelId);

    const model = new ClaudeProChatModel({
      token,
      modelId,
      maxTokens,
      temperature,
      topP,
      topK,
      extendedThinking,
      thinkingBudget,
      timeout,
      maxRetries,
      n8nContext: {
        addInputData: this.addInputData.bind(this),
        addOutputData: this.addOutputData.bind(this),
        connectionType: NodeConnectionTypes.AiLanguageModel as NodeConnectionType,
      },
    });

    return { response: model };
  }
}
