/*
                                              User asks a question
                                                      │
                                                      ▼
                                              ┌─────────────────────────────┐
                                              │  User Message               │
                                              │  "What is the time in       │
                                              │   India?"                   │
                                              └─────────────────────────────┘
                                                      │
                                                      ▼
                                              ┌─────────────────────────────┐
                                              │  ReAct Agent                │
                                              │  (Reasoning Step)           │
                                              │  - Understands intent       │
                                              │  - Matches intent to tool   │
                                              └─────────────────────────────┘
                                                      │
                                                      ▼
                                              ┌─────────────────────────────┐
                                              │  Tool Selection             │
                                              │  - Chooses get_time         │
                                              │  - Extracts city = "India"  │
                                              └─────────────────────────────┘
                                                      │
                                                      ▼
                                              ┌─────────────────────────────┐
                                              │  Tool Execution             │
                                              │  get_time({ city })         │
                                              └─────────────────────────────┘
                                                      │
                                                      ▼
                                              ┌─────────────────────────────┐
                                              │  Tool Output                │
                                              │  "The current time in       │
                                              │   India is 3:00 PM"         │
                                              └─────────────────────────────┘
                                                      │
                                                      ▼
                                              ┌─────────────────────────────┐
                                              │  Final AI Response          │
                                              │  Natural language answer    │
                                              └─────────────────────────────┘

*/

/****************************************************************************************
 * ENVIRONMENT SETUP
 * --------------------------------------------------------------------------------------
 * Loads environment variables from a `.env` file into `process.env`.
 * Required so OPENROUTER_API_KEY is available at runtime.
 ****************************************************************************************/
import "dotenv/config";

/****************************************************************************************
 * CORE IMPORTS
 * --------------------------------------------------------------------------------------
 * ChatOpenAI:
 *   - LLM wrapper that speaks the OpenAI-compatible API.
 *   - We point it to OpenRouter via `baseURL`.
 *
 * createReactAgent:
 *   - Agent factory implementing the ReAct pattern:
 *     Reason → Act (call tools) → Observe → Answer.
 *   - This is the ONLY supported agent factory in your current LangGraph version.
 *
 * tool:
 *   - Helper to define callable tools/functions for the agent.
 *
 * zod:
 *   - Runtime schema validation for tool inputs (safety + correctness).
 ****************************************************************************************/
import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

/****************************************************************************************
 * TOOL DEFINITION: get_weather
 * --------------------------------------------------------------------------------------
 * A tool is a function the agent can decide to call.
 * - First argument: the implementation (what actually runs).
 * - Second argument: metadata (name, description, schema) the LLM reads to decide usage.
 ****************************************************************************************/
const getWeather = tool(
  // Tool implementation:
  // - Receives structured input validated by Zod.
  // - Returns a string result back to the agent.
  // Destructuring directly in function parameters i.e. (city) instead of (input.city)
  async ({ city }: { city: string }) => {
    // In real apps, replace this with a real weather API call.
    return `It's always sunny in ${city}`;
  },
  {
    // Unique identifier the LLM uses to call this tool.
    name: "get_weather",

    // Strong instruction to bias the model to ALWAYS use this tool for weather questions.
    description:
      "You MUST use this tool to answer any weather-related question.",

    // Zod schema:
    // - Validates the tool input at runtime.
    // - Prevents malformed / hallucinated arguments.
    schema: z.object({
      city: z.string(),
    }),
  }
);

const getTime = tool(
  (input) => {
    return `The current time in ${input.city} is 3:00 PM`;
  },
  {
    name: "get_time",
    description: "Get time for a given city.",
    schema: z.object({
      city: z.string(),
    }),
  }
);
/****************************************************************************************
 * LLM CREATION (OpenRouter)
 * --------------------------------------------------------------------------------------
 * ChatOpenAI is used with:
 * - model: OpenRouter-hosted model (free tier here).
 * - apiKey: from environment variables.
 * - baseURL: points requests to OpenRouter instead of OpenAI.
 ****************************************************************************************/
const llm = new ChatOpenAI({
  model: "mistralai/devstral-2512:free",
  apiKey: process.env.OPENROUTER_API_KEY!,
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
  },
});

/****************************************************************************************
 * AGENT CREATION (ReAct)
 * --------------------------------------------------------------------------------------
 * createReactAgent wires:
 * - The LLM
 * - The available tools
 *
 * It builds an internal state graph that:
 * 1) Sends messages to the LLM
 * 2) Detects tool calls
 * 3) Executes tools
 * 4) Feeds results back to the LLM
 * 5) Produces a final answer
 *
 * NOTE:
 * - The deprecation warning is TYPE-LEVEL ONLY.
 * - There is no runtime replacement yet.
 * - Safe and correct to use.
 ****************************************************************************************/
// @ts-expect-deprecated — safe, intentional
const agent = createReactAgent({
  llm,
  tools: [getWeather, getTime],
});

/****************************************************************************************
 * AGENT INVOCATION
 * --------------------------------------------------------------------------------------
 * Messages are provided as tuples: [role, content]
 * - system: highest-priority instruction (forces tool usage)
 * - user: the actual user query
 ****************************************************************************************/
const result = await agent.invoke({
  messages: [
    // ["system", "You are a weather assistant."],
    // ["user", "What is the weather in New York?"],
    ["user", "What is the time in India?"],
    ["user", "What is the weather in US?"],
  ],
});

/****************************************************************************************
 * RESULT HANDLING
 * --------------------------------------------------------------------------------------
 * result.messages is the full conversation trace:
 * 1) System message
 * 2) User message
 * 3) Tool call
 * 4) Tool result
 * 5) Final assistant message
 *
 * We print the LAST message, which is the final answer to the user.
 ****************************************************************************************/
console.log(result.messages[result.messages.length - 1]?.content);
