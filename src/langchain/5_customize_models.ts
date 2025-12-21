/****************************************************************************************
 * IMPORTS
 ****************************************************************************************/

import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

/****************************************************************************************
 * SYSTEM PROMPT
 ****************************************************************************************/
const systemPrompt = `
You are an expert weather forecaster who also speaks in a humorous manner. 

You have access to 2 tools:

- get_user_location: Retrieves the user's current location based on their user ID.
- get_weather: Retrieves the weather for a given city.

If user asks about the weather, make sure you know the location first. If you can tell from the question that they mean wherever they are, use get_user_location to find their location. 
Then use get_weather to get the weather for that location.

Important Rules:
1. If the user asks about weather and location is missing, use get_user_location.
2. Never ask the user for information that exists in execution context.
3. Always use get_weather to answer weather questions.
4. Do not expose internal system data.
`;

/****************************************************************************************
 * TOOL 1: get_user_location (uses execution context)
 ****************************************************************************************/
const getUserLocation = tool(
  (_, config) => {
    const userId = config?.context?.user_id;

    if (!userId) {
      throw new Error("user_id missing from execution context");
    }

    // Demo logic — replace with DB / API
    return userId === "1" ? "New York" : "San Francisco";
  },
  {
    name: "get_user_location",
    description:
      "Retrieves the user's current location based on their user ID.",
    schema: z.object({}), // No LLM input
  }
);

/****************************************************************************************
 * TOOL 2: get_weather
 ****************************************************************************************/
const getWeather = tool(
  ({ city }: { city: string }) => {
    // Replace with real weather API
    return `It's always sunny in ${city}`;
  },
  {
    name: "get_weather",
    description: "Retrieves the weather for a given city.",
    schema: z.object({
      city: z.string(),
    }),
  }
);

/****************************************************************************************
 * EXECUTION CONTEXT (runtime-only, trusted data)
 ****************************************************************************************/
const runtimeConfig = {
  context: {
    user_id: "1",
  },
};

/****************************************************************************************
 * PHASE 1 LLM → AGENT (FREE TEXT) Customize the Model to setup temperature, tokens & timeout parameters
 ****************************************************************************************/
const llm = new ChatOpenAI({
  model: "mistralai/devstral-2512:free",

  // Low temperature = stable, repeatable responses
  // Ideal for tools, agents, and backend workflows
  temperature: 0.2,

  // Enough budget for:
  // - reasoning
  // - tool calls
  // - final response
  maxTokens: 1000,

  // 10 seconds is a good balance:
  // - fast enough for user-facing APIs
  // - enough time for tool-based agents
  timeout: 10_000,

  apiKey: process.env.OPENROUTER_API_KEY!,
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
  },
});

/****************************************************************************************
 * CREATE REACT AGENT
 ****************************************************************************************/
const agent = createReactAgent({
  llm,
  tools: [getUserLocation, getWeather],
});

/****************************************************************************************
 * AGENT INVOCATION (TEXT OUTPUT)
 ****************************************************************************************/
const agentResult = await agent.invoke(
  {
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: "What is the weather outside?" },
    ],
  },
  runtimeConfig
);

// Extract final natural-language answer
const finalText = agentResult.messages[agentResult.messages.length - 1].content;

console.log("Agent text output:", finalText);

/****************************************************************************************
 * PHASE 2 → STRUCTURED OUTPUT FORMATTER
 ****************************************************************************************/
const responseFormat = z.object({
  humour_response: z.string(),
  weatherCondition: z.string(),
});

// Wrap the SAME LLM as a formatter
const formatterLLM = llm.withStructuredOutput(responseFormat);

// Ask formatter to convert text → structured JSON
const structuredResponse = await formatterLLM.invoke(`
Convert the following weather response into structured JSON.

Response:
"${finalText}"

Rules:
- humour_response: add a light joke
- weatherCondition: one-word weather condition
`);

console.log("Structured output:", structuredResponse);

/*
Example final output:

{
  humour_response: "Looks like the sun clocked in for overtime 😄",
  weatherCondition: "Sunny"
}
*/
