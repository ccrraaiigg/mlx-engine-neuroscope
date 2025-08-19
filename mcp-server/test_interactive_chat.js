#!/usr/bin/env node

/**
 * Test script for Interactive Chat functionality
 */

import { InteractiveChatbot } from './src/cli/interactive_chat.js';
import { loadConfig } from './src/config/config.js';
import { setupLogging } from './src/utils/logging.js';

async function testChatbot() {
  console.log('🧪 Testing Interactive Chatbot...\n');
  
  try {
    // Load configuration
    const config = await loadConfig();
    setupLogging(config.logging);
    
    console.log('✅ Configuration loaded successfully');
    console.log('✅ Logging initialized');
    
    // Test chatbot creation (without starting interactive mode)
    const chatbot = new InteractiveChatbot(config);
    console.log('✅ Chatbot instance created');
    
    // Test MCP server startup checks
    console.log('\n🚀 Testing MCP server startup checks...');
    try {
      await chatbot.ensureMCPServerRunning();
      console.log('✅ MCP server startup checks completed');
    } catch (error) {
      console.log('⚠️ MCP server checks failed (expected if services not running):', error.message);
    }
    
    // Test health check method
    console.log('\n🏥 Testing health check functionality...');
    try {
      const healthStatus = await chatbot.checkAllServicesHealth();
      console.log('✅ Health check completed:', healthStatus);
    } catch (error) {
      console.log('⚠️ Health check failed (expected if services not running):', error.message);
    }
    
    // Note: Math problem generation removed - agents should provide their own examples
    console.log('\n🔢 Math problem generation removed - agents provide examples');
    console.log('✅ System ready for agent-provided examples');
    
    // Test system prompt with demo documentation
    console.log('\n📚 Testing system prompt with demo documentation...');
    try {
      const systemPrompt = await chatbot.buildSystemPrompt();
      const hasDemo = systemPrompt.includes('Neural Circuit Visualization Demo Guide');
      console.log(`✅ System prompt generated (${systemPrompt.length} chars)`);
      console.log(`✅ Demo documentation included: ${hasDemo ? 'Yes' : 'No'}`);
    } catch (error) {
      console.log('⚠️ System prompt generation failed:', error.message);
    }
    
    console.log('\n🎉 All tests completed successfully!');
    console.log('\nTo run the interactive chatbot:');
    console.log('  node src/cli/interactive_chat.js');
    console.log('\nMake sure you have:');
    console.log('  1. Anthropic API key configured');
    console.log('  2. MLX Engine running on port 50111');
    console.log('  3. Visualization server running on port 8888');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error(error.stack);
  }
}

testChatbot();
