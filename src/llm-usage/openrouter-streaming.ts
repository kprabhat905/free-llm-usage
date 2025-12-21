// This script uses the OpenRouter SDK to stream responses from a language model,
// showcasing real-time output for user queries.

import { OpenRouter } from "@openrouter/sdk";
import "dotenv/config";

// Asynchronous function to handle the streaming chat request
async function run() {
  // Initialize OpenRouter client with API key from environment
  const openrouter = new OpenRouter({
    apiKey: process.env.OPENROUTER_API_KEY!,
  });

  // Send a streaming chat request to the model
  const stream = await openrouter.chat.send({
    model: "mistralai/devstral-2512:free",
    messages: [
      {
        role: "user",
        content: "What is 2 + 2?",
      },
    ],
    stream: true, // Enable streaming for real-time response
  });

  // Iterate over the stream chunks and print content as it arrives
  for await (const chunk of stream) {
    const content = chunk.choices?.[0]?.delta?.content;
    if (content) {
      process.stdout.write(content); // Write directly to stdout for streaming effect
    }
  }

  // Add a newline after the stream ends
  console.log();
}

// Execute the run function and catch any errors
run().catch(console.error);
