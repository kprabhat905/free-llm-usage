// This script demonstrates how to use the OpenAI SDK to interact with a language model
// via OpenRouter, which provides access to various AI models including free tiers.

import OpenAI from 'openai';

async function run() {
// Initialize the OpenAI client with OpenRouter's base URL and API key from environment variables
const openai = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY!,
});

  // Create a chat completion with a simple math question
  const completion = await openai.chat.completions.create({
    model: "mistralai/devstral-2512:free",
    messages: [
      {
        "role": "user",
        "content": "What is 2 + 2?"
      }
    ]
  });

  // Log the response message from the AI
  console.log(completion.choices[0].message);

}

// Execute the run function and catch any errors
run().catch(console.error);