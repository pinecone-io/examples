import os
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentOutputParser, load_tools, initialize_agent
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish


class OutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS
    
    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            # this will work IF the text is a valid JSON with action and action_input
            response = parse_json_markdown(text)
            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                # this means the agent is finished so we call AgentFinish
                return AgentFinish({"output": action_input}, text)
            else:
                # otherwise the agent wants to use an action, so we call AgentAction
                return AgentAction(action, action_input, text)
        except Exception:
            # sometimes the agent will return a string that is not a valid JSON
            # often this happens when the agent is finished
            # so we just return the text as the output
            return AgentFinish({"output": text}, text)
    
    @property
    def _type(self) -> str:
        return "conversational_chat"


class Chatbot:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo-0613",
        openai_api_key: str = None,
        verbose: Optional[bool] = False
    ):
        self.model_name = model_name
        openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        assert openai_api_key is not None, "Please provide an OpenAI API key"
        # initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=0.3
        )
        # initialize memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=5, return_messages=True, output_key="output"
        )
        # initialize tools for agent
        self.tools = load_tools(["llm-math"], llm=self.llm)
        # initialize output parser for agent
        parser = OutputParser()
        # initialize agent
        self.agent = initialize_agent(
            agent="chat-conversational-react-description",
            tools=self.tools,
            llm=self.llm,
            verbose=verbose,
            early_stopping_method="generate",
            memory=self.memory,
            agent_kwargs={"output_parser": parser}
        )
        # update the prompt with custom system message
        sys_msg = """Assistant is a very helpful shopping assistant named Joey that helps users with any shopping task.

        Joey is very friendly and amicable, driving the conversation forward with the user's shopping plans while still making small talk.
        
        Joey is qualified to help with a wide range of tasks, but is not able to do any math related tasks."""
        new_prompt = self.agent.agent.create_prompt(
            system_message=sys_msg, tools=self.tools
        )
        self.agent.agent.llm_chain.prompt = new_prompt

    def _call(self, inputs):
        res = self.agent({'input': inputs})
        return res['output']
