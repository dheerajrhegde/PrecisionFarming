from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
import operator, os
from langchain_openai import ChatOpenAI
import AgentTools as tools
import pprint, uuid
from langchain_core.output_parsers import JsonOutputParser, MarkdownListOutputParser
import AgentTools as tools

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tool_list, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        #graph.add_node("end", self.parse_final_output)

        #graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: "end"})
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")

        graph.set_entry_point("llm")
        #graph.set_finish_point("end")

        self.graph = graph.compile()
        self.tool_list = {t.name: t for t in tool_list}
        self.model = model.bind_tools(tool_list)


    def call_openai(self, state: AgentState):
        #("Calling open AI")
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        #print("in exists action", result.tool_calls)
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            #print(f"Calling: {t}")
            result = self.tool_list[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        # print("Back to the model!")
        return {'messages': results}

    def parse_final_output(self, state: AgentState):
        print("Parsing final output")
        message = state['messages'][-1]  # | self.model.output_parser
        #response = JsonOutputParser(pydantic_object=tools.Guidance).parse(message.content)
        response = MarkdownListOutputParser().parse(message.content)
        return {'messages': [response]}


class PrecisionFarming:
    def __init__(self):
        self.model = ChatOpenAI(model='gpt-4o', openai_api_key=os.getenv("OPENAI_API_KEY"), )
        self.tool_list = [tools.decrease_ph, tools.get_weather_data,
                     tools.get_crop_info, tools.calculate_water_needed, tools.tackle_insect, tools.tackle_disease,
                          tools.increase_ph]
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

    def get_insights(self, soil_ph = 6.5, soil_moisture = 30, latitude = 35.41, longitude= -80.58,
                     area_acres = 10, crop = "Corn", insect = None, leaf = None):

        print("inset-->", insect, type(insect))
        print("leaf-->", leaf, type(leaf))

        if insect is not None: insect = tools.predict_insect(insect)
        if leaf is not None:
            if crop == "Corn":
                leaf = tools.predict_corn_leaf_disease(leaf)
            elif crop == "Cotton":
                leaf = tools.predict_cotton_leaf_disease(leaf)
            elif crop == "Soybean":
                leaf = tools.predict_soybean_leaf_disease(leaf)

        prompt = self.prompt.format(leaf=leaf,
                                    insect=insect,
                                    soil_ph=soil_ph,
                                    soil_moisture=soil_moisture,
                                    latitude=latitude,
                                    longitude=longitude,
                                    area_acres=area_acres,
                                    crop=crop)
        abot = Agent(self.model, self.tool_list, system=prompt)
        thread = {"configurable": {"thread_id": uuid.uuid4()}}
        question = "Give me your precision farming assessment"
        response = abot.graph.invoke(
            {"messages": [HumanMessage(content=[{"type": "text", "text": question}])], "thread": thread})
        return response['messages'][-1].content

if __name__ == "__main__":
    pf = PrecisionFarming()
    pprint.pprint(pf.get_insights())