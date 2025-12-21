/*
                                                              User asks a question
                                                                      │
                                                                      ▼
                                                              ┌───────────────────────────┐
                                                              │  User Message             │
                                                              │  "What is the weather     │
                                                              │   outside?"               │
                                                              └───────────────────────────┘
                                                                      │
                                                                      ▼
                                                              ┌───────────────────────────┐
                                                              │  ReAct Agent              │
                                                              │  (Reasoning Step)         │
                                                              │  - Needs location         │
                                                              │  - User didn’t provide it │
                                                              └───────────────────────────┘
                                                                      │
                                                                      ▼
                                                              ┌───────────────────────────┐
                                                              │  Tool 1: get_user_location│
                                                              │  - Reads context.user_id  │
                                                              │  - No user input needed   │
                                                              └───────────────────────────┘
                                                                      │
                                                                      ▼
                                                              ┌───────────────────────────┐
                                                              │  Tool Output              │
                                                              │  "New York"               │
                                                              └───────────────────────────┘
                                                                      │
                                                                      ▼
                                                              ┌───────────────────────────┐
                                                              │  Tool 2: get_weather      │
                                                              │  - Uses city = "New York" │
                                                              └───────────────────────────┘
                                                                      │
                                                                      ▼
                                                              ┌───────────────────────────┐
                                                              │  Tool Output              │
                                                              │  "It's always sunny..."   │
                                                              └───────────────────────────┘
                                                                      │
                                                                      ▼
                                                              ┌───────────────────────────┐
                                                              │  Final AI Response        │
                                                              │  "The weather outside is  │
                                                              │   sunny."                 │
                                                              └───────────────────────────┘

*/
/****************************************************************************************
 * AGENT + TOOL SETUP WITH USER CONTEXT
 * --------------------------------------------------------------------------------------
 * This file demonstrates:
 * 1. How to define tools
 * 2. How tools can use execution context (like user_id)
 * 3. How an agent chains multiple tools together
 * 4. How context flows through the agent → tool → agent loop
 ****************************************************************************************/

import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import "dotenv/config";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

/****************************************************************************************
 * TOOL 1: get_user_location
 * --------------------------------------------------------------------------------------
 * Purpose:
 * - Determines the user's location WITHOUT asking the user explicitly.
 * - Uses execution context (user_id) instead of LLM input.
 *
 * Key idea:
 * - Some tools depend on background data (context), not user messages.
 ****************************************************************************************/
const getUserLocation = tool(
  // FUNCTION SIGNATURE:
  // 1st argument (_) :
  //   - Input from the LLM.
  //   - This tool does not need any input, so it is ignored using '_'.
  //
  // 2nd argument (config):
  //   - Automatically provided by LangChain at execution time.
  //   - Contains context such as user_id, session data, metadata, etc.
  (_, config) => {
    // Read the user_id from the execution context
    const userId = config.context.user_id;

    // Demo logic:
    // - Hardcoded mapping for demonstration purposes
    // - In real systems, replace with:
    //   • database lookup
    //   • user profile service
    //   • IP / GPS-based lookup
    return userId === "1" ? "New York" : "San Francisco";
  },
  {
    // Tool name used internally by the LLM to decide which tool to call
    name: "get_user_location",

    // Description guides the LLM on WHEN this tool is useful
    description:
      "Retrieves the user's current location based on their user ID.",

    // Empty schema:
    // - Indicates NO input arguments are required from the LLM
    // - All data comes from execution context
    schema: z.object({}),
  }
);

/****************************************************************************************
 * TOOL 2: get_weather
 * --------------------------------------------------------------------------------------
 * Purpose:
 * - Retrieves weather information for a given city.
 * - Depends on structured input extracted from either:
 *   • user message
 *   • output of another tool (e.g., get_user_location)
 ****************************************************************************************/
const getWeather = tool(
  // Destructuring is used to directly access the 'city' field
  ({ city }: { city: string }) => {
    // Demo response:
    // - Replace with a real weather API call in production
    return `It's always sunny in ${city}`;
  },
  {
    // Name used by the LLM to reference this tool
    name: "get_weather",

    // Description tells the LLM this tool provides weather data
    description: "Retrieves the weather for a given city.",

    // Schema enforces that a city string MUST be provided
    schema: z.object({
      city: z.string(),
    }),
  }
);

/****************************************************************************************
 * EXECUTION CONTEXT
 * --------------------------------------------------------------------------------------
 * Context is external information passed to the agent at runtime.
 * - Not generated by the LLM
 * - Not visible to the user
 * - Used by tools that need personalization or session awareness
 ****************************************************************************************/
const config = {
  context: {
    // Identifies the current user
    user_id: "1",
  },
};

/****************************************************************************************
 * LLM CONFIGURATION (OpenRouter)
 * --------------------------------------------------------------------------------------
 * ChatOpenAI is used as a generic OpenAI-compatible client.
 * OpenRouter is specified via baseURL.
 ****************************************************************************************/
const llm = new ChatOpenAI({
  model: "mistralai/devstral-2512:free",
  apiKey: process.env.OPENROUTER_API_KEY!,
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
  },
});

/****************************************************************************************
 * AGENT CREATION (ReAct Pattern)
 * --------------------------------------------------------------------------------------
 * createReactAgent:
 * - Enables reasoning + tool usage
 * - Allows chaining tools together
 * - Manages decision-making automatically
 *
 * Tools registered:
 * - get_user_location
 * - get_weather
 ****************************************************************************************/
const agent = createReactAgent({
  llm,
  tools: [getUserLocation, getWeather],
});

/****************************************************************************************
 * AGENT INVOCATION
 * --------------------------------------------------------------------------------------
 * Flow:
 * 1. User asks a general question ("What is the weather outside?")
 * 2. Agent realizes location is needed
 * 3. Agent calls get_user_location (uses context.user_id)
 * 4. Agent receives city name
 * 5. Agent calls get_weather with that city
 * 6. Agent returns a final, user-friendly answer
 ****************************************************************************************/
const result = await agent.invoke(
  {
    messages: [
      {
        role: "user",
        content: "What is the weather outside?",
      },
    ],
  },
  config // Context passed separately from messages
);

// Log the full execution result (messages, tool calls, final answer)
console.log(result.messages[result.messages.length - 1].content);
