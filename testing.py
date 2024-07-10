import os
# from getpass import getpass
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import  AgentExecutor, create_react_agent
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
# from langchain_core.messages import HumanMessage
from langchain import hub

# Set environment variables
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENWEATHERMAP_API_KEY"] = ""

# Initialize the language model
llm = OpenAI(temperature=0)

# Define the modified template for the agent's prompt
template = '''
The user asks the question, first check the message history for relevant information.

Then, if the question is not about weather or wearing, provide a simple response using the language model (LLM).

If the question is about weather or wearing, use the following tools to answer the question {tools} and use the following format and provide detailed explanations:



Question: {input}
Action: If the question is about wearing/clothing/dressing or weather-related, then the action to take should be one of [{tool_names}]. If the question is not about weather or wearing, use a simple LLM response.
Action Input: The input to the action.
Observation: The result of the action.
... (Thought/Action/Action Input/Observation can repeat 2 times)
Thought: Summarize your observations and explain how you reached the conclusion.
Final Answer: The final answer to the original input question.

Begin!
Specific Query: Check the weather in the giving city/area with the information as today or tommorow and then decide what to wear base on that. Also note that you should give me exact clothing names in response like jeans, shalwar kameez, kurta pajama etc or any local clothing names for that area. Note if the question is not about weather or wearing don't use the agent/tool. Note first the check the history for the answers
Thought:{agent_scratchpad}
'''




# Create the prompt
prompt = PromptTemplate.from_template(template)


# Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/react")

# print("prompt checking ", prompt)

# Load the tools, including the OpenWeatherMap API
tools = load_tools(["openweathermap-api"], llm)


memory = ConversationBufferWindowMemory(return_messages=True, memory_key='chat_history', input_key='input', k=5)

# prompt1_system = "You are a helpful assistant."

# prompt1 = ChatPromptTemplate.from_messages([
#     ("system", prompt1_system),
#     MessagesPlaceholder(variable_name='chat_history'),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name='agent_scratchpad')
# ])

# Create the agent
agent_chain = create_react_agent(tools=tools, llm=llm, prompt=prompt)

# Prepare the input for the agent, including the 'intermediate_steps' key
# input_data = {
#     "input": "What's the weather like in Karachi? and what should I wear according to the weather in Karachi. Give me exact clothing names like jeans, shalwar kameez, kurta pajama etc.",
#     "intermediate_steps": ""
# }




def final_ans(input):
    agent_executor = AgentExecutor(agent=agent_chain, tools=tools, memory=memory, verbose=True)
    print("memory checking ", memory)
    final_answer = agent_executor.invoke({"input": input})
    return final_answer

# # Invoke the agent
# response = agent_chain.invoke(input_data)
# print(response)


# "What's the weather like in Karachi? and what should I wear according to the weather in Karachi. Give me exact clothing names like jeans, shalwar kameez, kurta pajama etc.