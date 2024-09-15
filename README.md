Purpose of this project is to build a precision engine that can help farmers automate their farming operations. And do so in a very optimal way. 

<img width="612" alt="image" src="https://github.com/user-attachments/assets/30268dfd-f121-43b4-b319-2b885098d775">

The visionn for the end result would be
- recommend the best time to plant the crops taking into account the climate requirements and the forecast of the best time to sell the crop
- Once planted, automate the aggregation of data
  - soil information from gauges in the farm land
  - weather information from weather APIs
  - insect and disease images from drones
- With real time data, make decisions on actions to take, define specific actions for the centralling controlled farm equipment

Current state of the project has built a RAG based core Precision Farming engine that 
- takes user inputs (did not have access to soil meters and drones),
- analyzes the specific of the crop and decides on the action plan
- gives the action plan and rationale as a narrative back to the farmer (did not have access to farm equipment)

<img width="612" alt="image" src="https://github.com/user-attachments/assets/bcb426a8-84d4-4d69-9d28-867b07f63714">

## Overview of the core engine

The core engine is developed using LangChain, LangGraph, and OpenAI. The engine takes a methodical approach to understanding the current state, analyzing it, and recommending a course of action. 
- Collect user input of location, crop, current soil moisture, insect image, leaf image
- gets 7 day weather forecast and recommends irrigation plan
- Predicts the insect in given image. Suggests actions to take based on information in crop guides available in vector stores
- Predicts the disease based on crop leaf image. Suggests action to take based on information in crop guides available in vector stores
- Get optimal PH and moisture levels. Suggests actions to take based on information in crop guides available in vector stores
- Searches the web incase relevent infromation is not available in the vector store
- Finally, puts its all toegther into a actionable plan for the farmer

## Technical details
The anchor for the graph is a function calling agentic workflow that uses Open AI and LangGraph. The graph has at its disposal few tools that it can decide to call based on the need. And once it has all the informationm, it puts together a structured markdown response to be given back to the user.

<img width="612" alt="image" src="https://github.com/user-attachments/assets/ad946711-f02e-4c7f-9142-d5a93c8bd757">

*** Key Decision: *** The controlling is an agentic tool based workflow with just the nodes to call tools and LLM. We decided to go with this approach instead of a well defined graph and nodes to ensure that the core graph can be chatty and refine the response as needed to meet the expectatios of the prompt. In contrary, the retrieval graph is well defined with specific nodes and conditional edges that takes a task from START to END. Retrieval graph was define in that way since we knew exactly how to get a well grounded intermediate response.  

### Tools

<p> decrease_ph and increase_ph - These are simple python functions annotated with @tool and does a predefined mathematical calculation on the about of chemicals to use to alter the PH to desired levels. <p>

get_weather_data - This uses the "weatherapi" API to get 7 day forecast for the location provided

calculate_water_needed - Simple python function that tells us how much water we need to get the soil moisture level to where we need it to be.

get_crop_info - generic funtion that uses the retrieval graph to answer questions that are not addressed by earlier defiend tools. Relies first on the crop production guides and then on web search

tackle_insect, tackle_disease - uses the retrieval graph to get needed information from the crop production guides that are chunked and stored in the vector database. Falls back on websearch if needed.

### Nodes in retreival graph

Retreive - Uses multi query translation to break down larger and complex queries into simple questions to do a vector search on. Uses metadata filtering to only get chunks from the guide that is related to the crop the farmer is growing. 

Context relevence - 

Web Search - 

Generate Response - 

Response Grounded - 

Transform Query - 

### Image Classification - Insect and Leaf (CNN using Tensorflow)

fine tuning of the ImageNet with softmax in last layer for multiclass classification. 

## Final response
<img width="1117" alt="image" src="https://github.com/user-attachments/assets/fb76a47c-2f11-4896-9921-7174af7a58bd">



