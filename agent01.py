from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# for Web search 
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

# Web search tool using DuckDuckGo
search_tool = Tool(
    name = "DuckDuckGo Search",
    func=DuckDuckGoSearchRun().run,
    description="Use this tool to perform web searches using DuckDuckGo. "
)

llm = OpenAI(
    temperature=0,
    # openai_api_key=openai_api_key,
    # model="gpt-3.5-turbo-16k",
    # max_tokens=2000
)

tools = [
    Tool(
        name="Search",
        func=search_tool.run,
        description="Use this tool to perform web searches using DuckDuckGo."
    )
]

# Initialize the agent with the LLM and tools
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True
)
# Run the agent with a sample query
query = "What is the capital of France?"
response = agent.run(query)
print(response)




