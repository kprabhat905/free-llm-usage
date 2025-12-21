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

  const completion = await openrouter.chat.send({
    model: "mistralai/devstral-2512:free",
    messages: [
      {
        role: "user",
        content: "What is 2 + 2?",
      },
    ],
    stream: false, // Enable streaming for real-time response
  });

  console.log(completion.choices[0].message.content);
}

// Execute the run function and catch any errors
run().catch(console.error);
