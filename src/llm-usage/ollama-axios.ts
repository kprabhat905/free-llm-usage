// This script uses LangChain's ChatOpenAI to query a local Ollama instance
// running the DeepSeek model, demonstrating integration with local AI setups.

import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";

// Configure the ChatOpenAI instance to connect to a local Ollama server
const llm = new ChatOpenAI({
  model: "deepseek-r1:1.5b",
  temperature: 0, // Set temperature to 0 for deterministic responses
  openAIApiKey: "ollama", // Dummy key as Ollama doesn't require authentication
  configuration: {
    baseURL: "http://localhost:11434/v1", // Ollama's default API endpoint
  },
});

// Asynchronous function to run the AI query
async function run() {
  // Invoke the model with a simple question and log the response content
  const res = await llm.invoke("What is 2 + 2?");
  console.log(res.content);
}

// Execute the run function
run();