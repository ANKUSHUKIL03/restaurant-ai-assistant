import express from "express";
import dotenv from "dotenv";
import path from "path";
import { z } from "zod";

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";

dotenv.config();

const port = 3000;
const app = express();

app.use(express.json());
app.use(express.static(path.join(path.resolve(), "public")));

const __dirname = path.resolve();

const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
  maxOutputTokens: 2048,
  temperature: 0.3,
  apiKey: process.env.GOOGLE_API_KEY,
});

const getMenuTool = new DynamicStructuredTool({
  name: "get-menu",
  description:
    "Use this tool whenever user asks about breakfast, lunch, or dinner menu.",
  schema: z.object({
    category: z
      .enum(["breakfast", "lunch", "dinner"])
      .describe("Menu category"),
    restaurant: z.string().optional().describe("Restaurant name if provided"),
  }),
  func: async ({ category }) => {
    const menus = {
      breakfast: ["pancakes", "omelette", "coffee"],
      lunch: ["burger", "fries", "soda"],
      dinner: ["steak", "salad", "wine"],
    };

    return `${category} menu items are: ${menus[category].join(", ")}`;
  },
});

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a restaurant menu assistant.
If the user asks about breakfast, lunch, or dinner, use the get-menu tool.
Do not ask for restaurant name unless the user specifically asks for a particular restaurant.`,
  ],
  ["human", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"),
]);

const agent = await createToolCallingAgent({
  llm: model,
  tools: [getMenuTool],
  prompt,
});

const executor = new AgentExecutor({
  agent,
  tools: [getMenuTool],
  verbose: true,
  maxIterations: 10,
  returnIntermediateSteps: true,
});

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.post("/api/chat", async (req, res) => {
  const userInput = req.body.input;
  console.log("UserInput:", userInput);
  try {
    const response = await executor.invoke({
      input: userInput,
    });

    console.log("Agent full Response:", response);

    return res.json({
      output: response.output || "No answer found.",
    });
  } catch (err) {
    console.log("Error during agent execution:", err);

    res.status(500).json({
      output: "Sorry, something went wrong. Please try again.",
    });
  }
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});