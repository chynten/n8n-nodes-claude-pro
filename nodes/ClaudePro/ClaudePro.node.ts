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

function openAiToolToAnthropic(tools: BindToolsInput[]): AnthropicTool[] {
  return tools.map((tool) => {
    const openAiTool = convertToOpenAITool(tool);
    return {
      name: openAiTool.function.name,
      description: openAiTool.function.description || '',
      input_schema: openAiTool.function.parameters || { type: 'object', properties: {} },
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
  extendedThinking: boolean;
  thinkingBudget?: number;
  tools?: AnthropicTool[];
  n8nContext?: N8nExecutionContext;
}

class ClaudeProChatModel extends BaseChatModel<ClaudeProCallOptions> {
  private token: string;
  private modelId: string;
  private maxTokens: number;
  private temperature: number;
  private extendedThinking: boolean;
  private thinkingBudget?: number;
  private boundTools?: AnthropicTool[];
  private n8nContext?: N8nExecutionContext;

  lc_serializable = false;

  constructor(fields: ClaudeProModelInput) {
    super({});
    this.token = fields.token;
    this.modelId = fields.modelId;
    this.maxTokens = fields.maxTokens;
    this.temperature = fields.temperature;
    this.extendedThinking = fields.extendedThinking;
    this.thinkingBudget = fields.thinkingBudget;
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
      extendedThinking: this.extendedThinking,
      thinkingBudget: this.thinkingBudget,
      tools: anthropicTools,
      n8nContext: this.n8nContext,
    }) as unknown as Runnable<BaseLanguageModelInput, AIMessageChunk, ClaudeProCallOptions>;
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    // Log input to n8n execution tracker
    let runIndex: number | undefined;
    if (this.n8nContext) {
      const inputPayload: INodeExecutionData[] = messages.map((msg) => ({
        json: { role: msg._getType(), content: msg.content },
      }));
      const result = this.n8nContext.addInputData(this.n8nContext.connectionType, [inputPayload]);
      runIndex = result.index;
    }

    try {
      const chatResult = await this._callApi(messages, options);

      // Log output to n8n execution tracker
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

  private async _callApi(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
  ): Promise<ChatResult> {
    const { system, filtered } = extractSystemMessage(messages);
    const apiMessages = convertMessages(filtered);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const body: Record<string, any> = {
      model: this.modelId,
      max_tokens: this.maxTokens,
      messages: apiMessages,
    };

    if (system) {
      body.system = system;
    }

    const tools = (options.tools && options.tools.length > 0) ? options.tools : this.boundTools;
    if (tools && tools.length > 0) {
      body.tools = tools;
    }

    if (this.extendedThinking && this.thinkingBudget) {
      body.thinking = { type: 'enabled', budget_tokens: this.thinkingBudget };
      body.temperature = 1;
    } else {
      body.temperature = this.temperature;
    }

    const headers = buildAuthHeaders(this.token);
    const https = require('https');

    const response = await new Promise<AnthropicResponse>((resolve, reject) => {
      const postData = JSON.stringify(body);
      const reqOptions = {
        hostname: 'api.anthropic.com',
        path: '/v1/messages',
        method: 'POST',
        headers: { ...headers, 'content-length': Buffer.byteLength(postData) },
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const req = https.request(reqOptions, (res: any) => {
        let rawData = '';
        res.on('data', (chunk: Buffer) => { rawData += chunk.toString(); });
        res.on('end', () => {
          if (res.statusCode && res.statusCode >= 400) {
            reject(new Error(`Anthropic API error (HTTP ${res.statusCode}): ${rawData}`));
            return;
          }
          try {
            resolve(JSON.parse(rawData) as AnthropicResponse);
          } catch {
            reject(new Error(`Failed to parse Anthropic response: ${rawData}`));
          }
        });
      });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      req.on('error', (error: any) => reject(error));
      req.write(postData);
      req.end();
    });

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
        return models.map((m) => ({
          name: m.display_name,
          value: m.id,
        }));
      },
    },
  };

  async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
    const credentials = await this.getCredentials('claudeProApi');
    const token = credentials.setupToken as string;

    const modelId = this.getNodeParameter('model', itemIndex) as string;
    const maxTokens = this.getNodeParameter('maxTokens', itemIndex, 4096) as number;
    const temperature = this.getNodeParameter('temperature', itemIndex, 1.0) as number;
    const extendedThinking = this.getNodeParameter('extendedThinking', itemIndex, false) as boolean;
    const thinkingBudget = extendedThinking
      ? (this.getNodeParameter('thinkingBudget', itemIndex, 10000) as number)
      : undefined;

    const model = new ClaudeProChatModel({
      token,
      modelId,
      maxTokens,
      temperature,
      extendedThinking,
      thinkingBudget,
      n8nContext: {
        addInputData: this.addInputData.bind(this),
        addOutputData: this.addOutputData.bind(this),
        connectionType: NodeConnectionTypes.AiLanguageModel as NodeConnectionType,
      },
    });

    return { response: model };
  }
}
