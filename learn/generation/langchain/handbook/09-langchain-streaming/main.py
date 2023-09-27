import os
from threading import Thread
from typing import Generator

import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from queue import Queue
from pydantic import BaseModel

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler

app = FastAPI()

class QueueCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue: Queue):
        self.queue = queue
        self.content: str = ""
        self.final_answer: bool = False

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.content += token
        # if we passed the final answer, we put tokens in queue
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ['"', "}"]:
                    self.queue.put(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""

# queue is our communication between threads
queue = Queue()
# we use this for signaling the end of the queue
job_done = "<END-STREAM>"

# initialize the agent
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
    model_name="gpt-3.5-turbo",
    streaming=True,  # ! important
    callbacks=[QueueCallbackHandler(queue)]  # ! important
)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
    output_key="output"
)
agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[],
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    return_intermediate_steps=False
)

# this handles the call to the agent, followed by job_done token
def call(query: str, queue: Queue) -> None:
    agent.run(query)
    queue.put(job_done)

# request input format
class Query(BaseModel):
    text: str

def stream(query: Query) -> Generator:
    """This function is what allows us to stream"""
    # move the agent task to another thread
    thread = Thread(target=call, args=(query.text, queue))
    thread.start()
    # on main thread, yield the results from queue to StreamingResponse
    while True:
        result = queue.get()
        if result is job_done:
            break
        yield result

@app.post("/chat")
async def chat(query: Query = Body(...)):
    return StreamingResponse(stream(query), media_type="text/event-stream")

@app.get("/health")
async def health():
    """Check the api is running"""
    return {"status": "🤙"}
    

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="localhost",
        port=8000,
        reload=True
    )