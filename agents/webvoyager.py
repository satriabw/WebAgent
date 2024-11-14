import asyncio
import platform
import base64
import re
from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from playwright.async_api import Page
from langchain_core.runnables import chain as chain_decorator, RunnablePassthrough, RunnableLambda
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv
import os
from IPython import display
from playwright.async_api import async_playwright

from agents.agent import Agent
from datatypes.datatypes import *

class WebVoyager(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.llm = ChatOpenAI(model="gpt-4o", max_tokens=4096, api_key=openai_api_key)
        self.prompt = hub.pull("wfh/web-voyager")
        self.agent = self.annotate | RunnablePassthrough.assign(
            prediction=self.format_descriptions | self.prompt | self.llm | StrOutputParser() | self.parse
        )
        self.graph_builder = StateGraph(AgentState)
        self.build_graph()

    async def click(self, state: AgentState):
        page = state["page"]
        click_args = state["prediction"]["args"]
        if click_args is None or len(click_args) != 1:
            return f"Failed to click bounding box labeled as number {click_args}"
        bbox_id = click_args[0]
        bbox_id = int(bbox_id)
        try:
            bbox = state["bboxes"][bbox_id]
        except Exception:
            return f"Error: no bbox for : {bbox_id}"
        x, y = bbox["x"], bbox["y"]
        await page.mouse.click(x, y)
        return f"Clicked {bbox_id}"

    async def type_text(self, state: AgentState):
        page = state["page"]
        type_args = state["prediction"]["args"]
        if type_args is None or len(type_args) != 2:
            return f"Failed to type in element from bounding box labeled as number {type_args}"
        bbox_id = type_args[0]
        bbox_id = int(bbox_id)
        bbox = state["bboxes"][bbox_id]
        x, y = bbox["x"], bbox["y"]
        text_content = type_args[1]
        await page.mouse.click(x, y)
        select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
        await page.keyboard.press(select_all)
        await page.keyboard.press("Backspace")
        await page.keyboard.type(text_content)
        await page.keyboard.press("Enter")
        return f"Typed {text_content} and submitted"

    async def scroll(self, state: AgentState):
        page = state["page"]
        scroll_args = state["prediction"]["args"]
        if scroll_args is None or len(scroll_args) != 2:
            return "Failed to scroll due to incorrect arguments."
        target, direction = scroll_args
        if target.upper() == "WINDOW":
            scroll_amount = 500
            scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
            await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
        else:
            scroll_amount = 200
            target_id = int(target)
            bbox = state["bboxes"][target_id]
            x, y = bbox["x"], bbox["y"]
            scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
            await page.mouse.move(x, y)
            await page.mouse.wheel(0, scroll_direction)
        return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"

    async def wait(self, state: AgentState):
        sleep_time = 5
        await asyncio.sleep(sleep_time)
        return f"Waited for {sleep_time}s."

    async def go_back(self, state: AgentState):
        page = state["page"]
        await page.go_back()
        return f"Navigated back a page to {page.url}."

    async def to_google(self, state: AgentState):
        page = state["page"]
        await page.goto("https://www.google.com/")
        return "Navigated to google.com."

    @chain_decorator
    async def mark_page(self, page):
        with open("mark_page.js") as f:
            mark_page_script = f.read()
        await page.evaluate(mark_page_script)
        for _ in range(10):
            try:
                bboxes = await page.evaluate("markPage()")
                break
            except Exception:
                asyncio.sleep(3)
        screenshot = await page.screenshot()
        await page.evaluate("unmarkPage()")
        return {
            "img": base64.b64encode(screenshot).decode(),
            "bboxes": bboxes,
        }

    async def annotate(self, state):
        marked_page = await self.mark_page.with_retry().ainvoke(state["page"], page=state["page"])
        return {**state, **marked_page}

    def format_descriptions(self, state):
        labels = []
        for i, bbox in enumerate(state["bboxes"]):
            text = bbox.get("ariaLabel") or ""
            if not text.strip():
                text = bbox["text"]
            el_type = bbox.get("type")
            labels.append(f'{i} (<{el_type}/>): "{text}"')
        bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
        return {**state, "bbox_descriptions": bbox_descriptions}

    def parse(self, text: str) -> dict:
        action_prefix = "Action: "
        if not text.strip().split("\n")[-1].startswith(action_prefix):
            return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
        action_block = text.strip().split("\n")[-1]
        action_str = action_block[len(action_prefix):]
        split_output = action_str.split(" ", 1)
        if len(split_output) == 1:
            action, action_input = split_output[0], None
        else:
            action, action_input = split_output
        action = action.strip()
        if action_input is not None:
            action_input = [inp.strip().strip("[]") for inp in action_input.strip().split(";")]
        return {"action": action, "args": action_input}

    def update_scratchpad(self, state: AgentState):
        old = state.get("scratchpad")
        if old:
            txt = old[0].content
            last_line = txt.rsplit("\n", 1)[-1]
            step = int(re.match(r"\d+", last_line).group()) + 1
        else:
            txt = "Previous action observations:\n"
            step = 1
        txt += f"\n{step}. {state['observation']}"
        return {**state, "scratchpad": [SystemMessage(content=txt)]}

    def build_graph(self):
        self.graph_builder.add_node("agent", self.agent)
        self.graph_builder.add_edge(START, "agent")
        self.graph_builder.add_node("update_scratchpad", self.update_scratchpad)
        self.graph_builder.add_edge("update_scratchpad", "agent")

        tools = {
            "Click": self.click,
            "Type": self.type_text,
            "Scroll": self.scroll,
            "Wait": self.wait,
            "GoBack": self.go_back,
            "Google": self.to_google,
        }

        for node_name, tool in tools.items():
            self.graph_builder.add_node(
                node_name,
                RunnableLambda(tool) | (lambda observation: {"observation": observation}),
            )
            self.graph_builder.add_edge(node_name, "update_scratchpad")

        self.graph_builder.add_conditional_edges("agent", self.select_tool)
        self.graph = self.graph_builder.compile()

    def select_tool(self, state: AgentState):
        action = state["prediction"]["action"]
        if "ANSWER" in action:
            return END
        if action == "retry":
            return "agent"
        return action

    async def call_agent(self, question: str, page, max_steps: int = 15):
        event_stream = self.graph.astream(
            {
                "page": page,
                "input": question,
                "scratchpad": [],
            },
            {
                "recursion_limit": max_steps,
            },
        )
        final_answer = None
        steps = []
        async for event in event_stream:
            if "agent" not in event:
                continue
            pred = event["agent"].get("prediction") or {}
            action = pred.get("action")
            action_input = pred.get("args")
            steps.append(f"{len(steps) + 1}. {action}: {action_input}")
            print("\n".join(steps))
            if "ANSWER" in action:
                final_answer = action_input[0]
                break
        return final_answer
