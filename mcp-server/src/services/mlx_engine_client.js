/**
 * MLX Engine REST API Client
 * 
 * Provides a client interface to communicate with the MLX Engine REST API
 * for model loading, activation capture, and analysis operations.
 */

import { getLogger } from '../utils/logging.js';

export class MLXEngineClient {
  constructor(config) {
    this.config = config;
    this.logger = getLogger('MLXEngineClient');
    this.baseUrl = config.apiUrl;
    this.timeout = config.timeout || 30000;
    this.retryAttempts = config.retryAttempts || 3;
    this.apiKey = config.apiKey;
    
    // Connection state
    this.isConnected = false;
    this.lastHealthCheck = null;
  }

  /**
   * Makes an HTTP request to the MLX Engine API
   * @param {string} endpoint - API endpoint path
   * @param {object} options - Request options
   * @returns {Promise<object>} Response data
   */
  async makeRequest(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const requestOptions = {
      method: options.method || 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    // Add API key if configured
    if (this.apiKey) {
      requestOptions.headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    // Add request body if provided
    if (options.body && typeof options.body === 'object') {
      requestOptions.body = JSON.stringify(options.body);
    }

    this.logger.debug(`Making request to ${url}`);

    try {
      const response = await fetch(url, requestOptions);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      this.logger.debug(`Request successful: ${endpoint}`);
      return data;
      
    } catch (error) {
      this.logger.error(`Request failed: ${endpoint} - ${error.message}`);
      throw error;
    }
  }

  /**
   * Checks the health status of the MLX Engine
   * @returns {Promise<object>} Health status
   */
  async checkHealth() {
    try {
      const health = await this.makeRequest('/health');
      this.isConnected = true;
      this.lastHealthCheck = new Date();
      return health;
    } catch (error) {
      this.isConnected = false;
      throw error;
    }
  }

  /**
   * Loads a model in the MLX Engine
   * @param {string} modelId - Model identifier
   * @param {object} options - Loading options
   * @returns {Promise<object>} Load result
   */
  async loadModel(modelId, options = {}) {
    this.logger.info(`Loading model: ${modelId}`);
    
    const requestBody = {
      model_path: `/Users/craig/.lmstudio/models/nightmedia/gpt-oss-20b-q5-hi-mlx`,
      model_id: modelId,
      ...options,
    };

    return await this.makeRequest('/v1/models/load', {
      method: 'POST',
      body: requestBody,
    });
  }

  /**
   * Creates activation hooks for capturing model internals
   * @param {Array} hookSpecs - Hook specifications
   * @returns {Promise<object>} Hook creation result
   */
  async createHooks(hookSpecs) {
    this.logger.info(`Creating ${hookSpecs.length} activation hooks`);
    
    // Convert hookSpecs to the format expected by the real API
    // Create separate hooks for each component in each layer
    const hooks = [];
    for (const spec of hookSpecs) {
      for (const component of spec.components) {
        if (component === 'attention') {
          hooks.push({
            layer_name: `model.layers.${spec.layer}.self_attn`,
            component: 'attention',
            hook_id: `layer_${spec.layer}_attention`,
            capture_input: false,
            capture_output: true
          });
        } else if (component === 'mlp') {
          hooks.push({
            layer_name: `model.layers.${spec.layer}.mlp`,
            component: 'mlp',
            hook_id: `layer_${spec.layer}_mlp`,
            capture_input: false,
            capture_output: true
          });
        }
      }
    }
    
    const requestBody = {
      hooks: hooks,
    };

    return await this.makeRequest('/v1/activations/hooks', {
      method: 'POST',
      body: requestBody,
    });
  }

  /**
   * Generates text with activation capture
   * @param {string} prompt - Input prompt
   * @param {object} options - Generation options
   * @returns {Promise<object>} Generation result with activations
   */
  async generateWithActivations(prompt, options = {}) {
    this.logger.info(`Generating with activations for prompt: "${prompt.substring(0, 50)}..."`);
    
    // Use explicit activation hooks for reliable capture
    const activation_hooks = [
      { layer_name: 'model.layers.0.self_attn', component: 'attention', hook_id: 'layer_0_attention' },
      { layer_name: 'model.layers.0.mlp', component: 'mlp', hook_id: 'layer_0_mlp' },
      { layer_name: 'model.layers.1.self_attn', component: 'attention', hook_id: 'layer_1_attention' },
      { layer_name: 'model.layers.1.mlp', component: 'mlp', hook_id: 'layer_1_mlp' }
    ];
    
    const requestBody = {
      messages: [{ role: 'user', content: prompt }],
      max_tokens: options.max_tokens || 50,
      temperature: options.temperature || 0.7,
      activation_hooks: activation_hooks,
      ...options,
    };

    return await this.makeRequest('/v1/chat/completions/with_activations', {
      method: 'POST',
      body: requestBody,
    });
  }

  /**
   * Analyzes mathematical reasoning circuits
   * @param {string} prompt - Mathematical prompt
   * @param {object} options - Analysis options
   * @returns {Promise<object>} Analysis results
   */
  async analyzeMath(prompt, options = {}) {
    this.logger.info(`Analyzing mathematical reasoning for: "${prompt}"`);
    
    // First generate with activations
    const generation = await this.generateWithActivations(prompt, {
      max_tokens: options.max_tokens || 100,
      temperature: options.temperature || 0.1,
    });

    // Then analyze the captured activations
    const analysisBody = {
      activations: generation.activations,
      analysis_type: 'mathematical_reasoning',
      ...options,
    };

    return await this.makeRequest('/analyze/math', {
      method: 'POST',
      body: analysisBody,
    });
  }

  /**
   * Analyzes attention patterns
   * @param {string} prompt - Input prompt
   * @param {Array} layers - Layers to analyze
   * @returns {Promise<object>} Attention analysis
   */
  async analyzeAttention(prompt, layers = [], scope = 'layer_level') {
    this.logger.info(`Analyzing attention patterns for layers: ${layers.join(', ')} with scope: ${scope}`);
    
    const analysisBody = {
      prompt: prompt,
      layers: layers.length > 0 ? layers : undefined,
      scope: scope,
    };

    return await this.makeRequest('/v1/analyze/attention', {
      method: 'POST',
      body: analysisBody,
    });
  }

  /**
   * Analyzes factual recall circuits
   * @param {string} factualPrompt - Factual query
   * @param {object} options - Analysis options
   * @returns {Promise<object>} Factual analysis
   */
  async analyzeFactual(factualPrompt, options = {}) {
    this.logger.info(`Analyzing factual recall for: "${factualPrompt}"`);
    
    const generation = await this.generateWithActivations(factualPrompt, {
      max_tokens: options.max_tokens || 50,
      temperature: 0.0, // Deterministic for factual queries
    });

    const analysisBody = {
      activations: generation.activations,
      analysis_type: 'factual_recall',
      query: factualPrompt,
      ...options,
    };

    return await this.makeRequest('/analyze/factual', {
      method: 'POST',
      body: analysisBody,
    });
  }

  /**
   * Tracks residual stream flow
   * @param {string} prompt - Input prompt
   * @param {object} options - Tracking options
   * @returns {Promise<object>} Residual stream analysis
   */
  async trackResidualStream(prompt, options = {}) {
    this.logger.info(`Tracking residual stream for: "${prompt}"`);
    
    const generation = await this.generateWithActivations(prompt, {
      capture_residual_stream: true,
      ...options,
    });

    // Extract residual stream data from activations
    const residualData = generation.activations || {};
    
    const analysisBody = {
      residual_data: residualData,
      analysis_type: 'residual_flow',
    };

    return await this.makeRequest('/analyze/residual', {
      method: 'POST',
      body: analysisBody,
    });
  }

  /**
   * Exports data in NeuroScope format
   * @param {object} analysisData - Analysis data to export
   * @param {string} format - Export format
   * @returns {Promise<object>} Export result
   */
  async exportNeuroScope(analysisData, format = 'smalltalk') {
    this.logger.info(`Exporting to NeuroScope format: ${format}`);
    
    const requestBody = {
      data: analysisData,
      format: format,
      export_type: 'neuroscope',
    };

    return await this.makeRequest('/export/neuroscope', {
      method: 'POST',
      body: requestBody,
    });
  }

  /**
   * Gets the current status of the MLX Engine
   * @returns {Promise<object>} Status information
   */
  async getStatus() {
    return await this.makeRequest('/status');
  }

  /**
   * Lists available models
   * @returns {Promise<object>} Available models
   */
  async listModels() {
    return await this.makeRequest('/models');
  }

  /**
   * Gets information about a specific model
   * @param {string} modelId - Model identifier
   * @returns {Promise<object>} Model information
   */
  async getModelInfo(modelId) {
    return await this.makeRequest(`/models/${modelId}`);
  }

  /**
   * Validates the connection to MLX Engine
   * @returns {Promise<boolean>} True if connected and healthy
   */
  async validateConnection() {
    try {
      await this.checkHealth();
      return true;
    } catch (error) {
      this.logger.warn(`Connection validation failed: ${error.message}`);
      return false;
    }
  }

  /**
   * Gets connection statistics
   * @returns {object} Connection stats
   */
  getConnectionStats() {
    return {
      isConnected: this.isConnected,
      baseUrl: this.baseUrl,
      lastHealthCheck: this.lastHealthCheck,
      timeout: this.timeout,
      retryAttempts: this.retryAttempts,
    };
  }
}

