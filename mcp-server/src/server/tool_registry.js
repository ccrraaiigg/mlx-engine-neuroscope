/**
 * Tool Registry for MCP Server
 * Manages tool registration, discovery, and validation
 */

// MCP validation simplified
import { getLogger } from '../utils/logging.js';

export class ToolRegistry {
  constructor() {
    this.tools = new Map();
    this.logger = getLogger('ToolRegistry');
  }

  /**
   * Registers a new tool with the registry
   * @param {object} tool - Tool definition with name, description, inputSchema, and handler
   * @throws {Error} If tool validation fails or tool already exists
   */
  registerTool(tool) {
    try {
      // Basic tool validation
      if (!tool.name || !tool.description || !tool.handler) {
        throw new Error('Tool must have name, description, and handler');
      }
      const validatedTool = tool;

      // Check if tool already exists
      if (this.tools.has(validatedTool.name)) {
        throw new Error(`Tool '${validatedTool.name}' is already registered`);
      }

      // Ensure handler is a function
      if (typeof tool.handler !== 'function') {
        throw new Error(`Tool '${validatedTool.name}' must have a handler function`);
      }

      // Store the complete tool definition including handler
      this.tools.set(validatedTool.name, {
        ...validatedTool,
        handler: tool.handler,
      });

      this.logger.info(`Registered tool: ${validatedTool.name}`);
    } catch (error) {
      this.logger.error(`Failed to register tool: ${error.message}`);
      throw error;
    }
  }

  /**
   * Registers multiple tools at once
   * @param {Array<object>} tools - Array of tool definitions
   */
  registerTools(tools) {
    for (const tool of tools) {
      this.registerTool(tool);
    }
  }

  /**
   * Gets a tool by name
   * @param {string} name - Tool name
   * @returns {object|null} Tool definition or null if not found
   */
  getTool(name) {
    return this.tools.get(name) || null;
  }

  /**
   * Gets all registered tools
   * @returns {Array<object>} Array of tool definitions (without handlers)
   */
  getAllTools() {
    return Array.from(this.tools.values()).map((tool) => ({
      name: tool.name,
      description: tool.description,
      inputSchema: tool.inputSchema,
      outputSchema: tool.outputSchema,
    }));
  }

  /**
   * Gets tools by category (based on name prefix)
   * @param {string} category - Category prefix (e.g., 'core', 'mlx', 'advanced')
   * @returns {Array<object>} Array of tools in the category
   */
  getToolsByCategory(category) {
    return this.getAllTools().filter((tool) => tool.name.startsWith(`${category}_`));
  }

  /**
   * Checks if a tool exists
   * @param {string} name - Tool name
   * @returns {boolean} True if tool exists
   */
  hasTool(name) {
    return this.tools.has(name);
  }

  /**
   * Validates tool input parameters against the tool's schema
   * @param {string} toolName - Name of the tool
   * @param {object} params - Input parameters to validate
   * @returns {Promise<object>} Validated parameters
   * @throws {Error} If validation fails
   */
  async validateToolInput(toolName, params) {
    const tool = this.getTool(toolName);
    if (!tool) {
      throw new Error(`Tool '${toolName}' not found`);
    }

    try {
      // Use Zod to validate against the input schema
      const schema = await this.createZodSchemaFromJSON(tool.inputSchema);
      return schema.parse(params);
    } catch (error) {
      throw new Error(`Input validation failed for tool '${toolName}': ${error.message}`);
    }
  }

  /**
   * Validates tool output against the tool's output schema (if defined)
   * @param {string} toolName - Name of the tool
   * @param {any} output - Output to validate
   * @returns {Promise<any>} Validated output
   * @throws {Error} If validation fails
   */
  async validateToolOutput(toolName, output) {
    const tool = this.getTool(toolName);
    if (!tool || !tool.outputSchema) {
      return output; // No output schema defined, skip validation
    }

    try {
      const schema = await this.createZodSchemaFromJSON(tool.outputSchema);
      return schema.parse(output);
    } catch (error) {
      throw new Error(`Output validation failed for tool '${toolName}': ${error.message}`);
    }
  }

  /**
   * Executes a tool with the given parameters
   * @param {string} toolName - Name of the tool to execute
   * @param {object} params - Parameters to pass to the tool
   * @returns {Promise<any>} Tool execution result
   * @throws {Error} If tool not found or execution fails
   */
  async executeTool(toolName, params = {}) {
    const tool = this.getTool(toolName);
    if (!tool) {
      throw new Error(`Tool '${toolName}' not found`);
    }

    try {
      // Validate input parameters
      const validatedParams = await this.validateToolInput(toolName, params);

      this.logger.info(`Executing tool: ${toolName}`);
      const startTime = Date.now();

      // Execute the tool handler
      const result = await tool.handler(validatedParams);

      const executionTime = Date.now() - startTime;
      this.logger.info(`Tool '${toolName}' completed in ${executionTime}ms`);

      // Validate output if schema is defined
      const validatedResult = await this.validateToolOutput(toolName, result);

      return validatedResult;
    } catch (error) {
      this.logger.error(`Tool execution failed for '${toolName}': ${error.message}`);
      throw error;
    }
  }

  /**
   * Converts a JSON Schema to a Zod schema for validation
   * @param {object} jsonSchema - JSON Schema object
   * @returns {Promise<object>} Zod schema
   */
  async createZodSchemaFromJSON(jsonSchema) {
    // This is a simplified implementation
    // In a full implementation, you'd need to handle all JSON Schema features
    const { z } = await import('zod');

    if (jsonSchema.type !== 'object') {
      throw new Error('Only object schemas are supported');
    }

    const shape = {};
    const required = jsonSchema.required || [];

    for (const [key, prop] of Object.entries(jsonSchema.properties || {})) {
      let propSchema = await this.createZodPropertySchema(prop);

      // If field is not required and doesn't have a default, make it optional
      if (!required.includes(key) && prop.default === undefined) {
        propSchema = propSchema.optional();
      }

      shape[key] = propSchema;
    }

    let schema = z.object(shape);

    if (jsonSchema.additionalProperties === false) {
      schema = schema.strict();
    }

    return schema;
  }

  /**
   * Creates a Zod schema for a JSON Schema property
   * @param {object} prop - JSON Schema property definition
   * @returns {object} Zod schema for the property
   */
  async createZodPropertySchema(prop) {
    const { z } = await import('zod');

    let schema;

    switch (prop.type) {
      case 'string':
        schema = z.string();
        if (prop.minLength) schema = schema.min(prop.minLength);
        if (prop.maxLength) schema = schema.max(prop.maxLength);
        if (prop.pattern) schema = schema.regex(new RegExp(prop.pattern));
        if (prop.enum) schema = z.enum(prop.enum);
        break;

      case 'number':
        schema = z.number();
        if (prop.minimum !== undefined) schema = schema.min(prop.minimum);
        if (prop.maximum !== undefined) schema = schema.max(prop.maximum);
        break;

      case 'integer':
        schema = z.number().int();
        if (prop.minimum !== undefined) schema = schema.min(prop.minimum);
        if (prop.maximum !== undefined) schema = schema.max(prop.maximum);
        break;

      case 'boolean':
        schema = z.boolean();
        break;

      case 'array':
        schema = z.array(z.any());
        if (prop.items) {
          schema = z.array(await this.createZodPropertySchema(prop.items));
        }
        break;

      case 'object':
        if (prop.properties) {
          const shape = {};
          for (const [key, subProp] of Object.entries(prop.properties)) {
            shape[key] = await this.createZodPropertySchema(subProp);
          }
          schema = z.object(shape);
        } else {
          schema = z.record(z.any());
        }
        break;

      default:
        schema = z.any();
    }

    // Handle default values by making the field optional with a default
    if (prop.default !== undefined) {
      schema = schema.optional().default(prop.default);
    }

    return schema;
  }

  /**
   * Unregisters a tool
   * @param {string} name - Tool name to unregister
   * @returns {boolean} True if tool was removed, false if not found
   */
  unregisterTool(name) {
    const removed = this.tools.delete(name);
    if (removed) {
      this.logger.info(`Unregistered tool: ${name}`);
    }
    return removed;
  }

  /**
   * Clears all registered tools
   */
  clear() {
    const count = this.tools.size;
    this.tools.clear();
    this.logger.info(`Cleared ${count} tools from registry`);
  }

  /**
   * Gets registry statistics
   * @returns {object} Registry statistics
   */
  getStats() {
    const tools = this.getAllTools();
    const categories = {};

    for (const tool of tools) {
      const category = tool.name.split('_')[0];
      categories[category] = (categories[category] || 0) + 1;
    }

    return {
      totalTools: tools.length,
      categories,
      toolNames: tools.map((t) => t.name),
    };
  }
}
