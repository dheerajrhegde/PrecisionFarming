import os
import operator
import pprint
import uuid
from typing import TypedDict, Annotated, Callable, Optional

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser, MarkdownListOutputParser
from langgraph.graph import StateGraph, END
import AgentTools as tools

AZURE_DEPLOYMENT="gpt-4o"
API_VERSION="2024-05-01-preview"
AZURE_ENDPOINT="https://agtech-llm-openai.openai.azure.com"
API_KEY="5366f9c0121f4852afeb69388c2aff3a"

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    status_callback: Optional[Callable]

class Agent:
    def __init__(self, model, tool_list, system="", status_callback=None):
        self.system = system
        self.status_callback = status_callback
        
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")

        self.graph = graph.compile()
        self.tool_list = {t.name: t for t in tool_list}
        self.model = model.bind_tools(tool_list)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        total_tools = len(tool_calls)
        
        for idx, t in enumerate(tool_calls, 1):
            if self.status_callback:
                progress = (idx / total_tools) * 100
                self.status_callback(f"Using {t['name']}", progress, t['name'])
            
            result = self.tool_list[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        
        return {'messages': results}

class PrecisionFarming:
    def __init__(self):
        self.model = AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOYMENT,
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=API_KEY)
        
        self.tool_list = [
            tools.decrease_ph, 
            tools.get_weather_data,
            tools.get_crop_info, 
            tools.calculate_water_needed, 
            tools.tackle_insect, 
            tools.tackle_disease,
            tools.increase_ph
        ]
        
        self.model.bind_tools(self.tool_list)
        self.prompt = """
            You are an Expert framing assistant. You will be given the following information
            Soil PH: {soil_ph}
            Soil Moisture: {soil_moisture}
            Latitude: {latitude}
            Longitude: {longitude}
            Area (acres): {area_acres}
            Crop: {crop}
            Insect Name: {insect}
            Leaf Disease Name: {leaf}
        
            Based on this information, get additional data and analyze
            1. Get the weather data for the location and find out how much rain is expected today and next few days
            2. Get the expected soil PH, soil moisture for the crop
            3. Calculate the amount of water the field will get over few days based on weather precipitation date
            4. Identify the action needed to get the PH to the desired level/range
            5. Identify the action needed to get the moisture to the desired level/range
            6. Identify the required fertilizers needed for the crop. Find the moisture level and weather condition ideal for spraying fertilizers
        
            Based on tool call results, provide a precision farming assessment in below format. Please do not use any other 
            information other than what was provided by the tools. Do not halunicate. Base your response on the data provided.
            Watering plan:
                Tell the farmer what the expected moisture percent
                Tell the farmer how much rain is expected today and coming days
                Based on that, tell the farmer how much water to give to the crops
            PH Control Plan:
                Tell the farmer what the ideal PH is for the crop
                Tell the farmer what action to take to get the PH to the desired level
                Call out the weather conditions you have considered while recommending the PH control time    
            Fertilizer plan:
                Tell the farmer what fertilizer is best for the crop and ideal weather conditions when to put the fertilizer
                Recommend when to fertilize the field
                Call out the weather conditions you have considered while recommending the fertilizing time
            Insect Control Plan:
                Get insights on how to remediate the insect infestation effecting the crop. Take into account the crop, the insect, weather, watering plan
                Tell the farmer what insect is in the image
                Tell the farmer what action to take to control the insect
                Explain your rationale for the action plan. What data did you consider to come up with the plan?
                explain your reasoning for the timing. Provide reference to the weather and moisture levels and you used it in your reasoning
            Leaf Disease Control Plan:
                Get insights on how to remediate the disease effecting the crop. Take into account the crop, the disease, weather, watering plan
                Tell the farmer what the disease is in the leaf image
                Create a date by date action plan to remediate the disease? Focus should be on remediation and not prevention. Remeber, framer already has the problem
                Explain your rationale for the action plan. What data did you consider to come up with the plan?
                explain your reasoning for the timing. Provide reference to the weather and moisture levels and you used it in your reasoning
        """

    def get_insights(self, soil_ph=6.5, soil_moisture=30, latitude=35.41, longitude=-80.58,
                    area_acres=10, crop="Corn", insect=None, leaf=None, status_callback=None):
        try:
            # if status_callback:
            #     status_callback("Starting analysis...", 0, "initial")

            # Process images if provided
            if insect is not None:
                if status_callback:
                    status_callback("Identifying The Insect/Pest...", 10, "Insect Processing")
                insect = tools.predict_insect(insect)
                status_callback("Identified The Insect/Pest.", 100, "Insect Processing")

            if leaf is not None:
                if status_callback:
                    status_callback("Identifying The Leaf...", 20, "Processing Leaf Image")
                if crop == "Corn":
                    leaf = tools.predict_corn_leaf_disease(leaf)
                elif crop == "Cotton":
                    leaf = tools.predict_cotton_leaf_disease(leaf)
                elif crop == "Soybean":
                    leaf = tools.predict_soybean_leaf_disease(leaf)
                status_callback("Identified The Leaf.", 100, "Processing Leaf Image")

            # Format prompt with inputs
            prompt = self.prompt.format(
                leaf=leaf,
                insect=insect,
                soil_ph=soil_ph,
                soil_moisture=soil_moisture,
                latitude=latitude,
                longitude=longitude,
                area_acres=area_acres,
                crop=crop
            )

            # Create agent with status tracking
            abot = Agent(self.model, self.tool_list, system=prompt, status_callback=status_callback)
            thread = {"configurable": {"thread_id": uuid.uuid4()}}
            question = "Give me your precision farming assessment"

            if status_callback:
                status_callback("Analyzing farm conditions...", 30, "Analysis")

            # Get farming assessment
            response = abot.graph.invoke({
                "messages": [HumanMessage(content=[{
                    "type": "text", 
                    "text": question}])],
                "thread": thread
            })

            if status_callback:
                status_callback("Analysis complete!", 100, "Analysis")

            return response['messages'][-1].content

        except Exception as e:
            if status_callback:
                status_callback(f"Error: {str(e)}", -1, "error")
            raise e

if __name__ == "__main__":
    pf = PrecisionFarming()
    pprint.pprint(pf.get_insights())