import os
from typing import TypeVar

import numpy as np
import requests
from keras.preprocessing import image
import keras
from langchain.agents import tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate

from RetrievalGraph import RetrievalGraph

# Type variable for PIL image
ImageBin = TypeVar('PIL.Image.Image')

# Model for input schemas
class InsectCropPlan(BaseModel):
    insect: str = Field(..., description="Insect to address")
    crop: str = Field(..., description="Crop to protect")


class ImagePath(BaseModel):
    image_path: str = Field(..., description="Path of the image to check for disease")


class PImage(BaseModel):
    img: ImageBin = Field(..., description="Image in binary form of PIL.Image.Image")


class Location(BaseModel):
    latitude: str = Field(..., description="The latitude of the location to get the weather for")
    longitude: str = Field(..., description="The longitude of the location to get the weather for")


class ImageNumpy(BaseModel):
    image_np: str = Field(..., description="A numpy array of the image")


class CropName(BaseModel):
    crop_name: str = Field(..., description="Name of the crop to get information for")


class CropQuestion(BaseModel):
    crop_question: str = Field(..., description="Question on the crop")
    crop: str = Field(..., description="Name of the crop to get information for")


class Guidance(BaseModel):
    topic: str = Field(..., description="Topic heading for your response, e.g. 'Watering plan'")
    guidance: str = Field(..., description="Guidance to give in your response for the topic")


# Load environment variable for weather API
weather_api_key = "1fa57cfecf054ea1974113726241310"

# Load models
reconstructed_model_soybean_leaf = keras.models.load_model("models/leaf.soybean.mobilenetv3large.keras")
reconstructed_model_cotton_leaf = keras.models.load_model("models/leaf.cotton.mobilenetv3large.keras")
reconstructed_model_corn_leaf = keras.models.load_model("models/leaf.corn.mobilenetv3large.keras")
reconstructed_model_insect = keras.models.load_model("models/insect.mobilenetv3large.keras")

retrieval_graph = RetrievalGraph()

#@tool(args_schema=PImage)
def predict_soybean_leaf_disease(img):
    """ Tell whether the soybean leaf has a disease or is healthy """
    #img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    classes = reconstructed_model_soybean_leaf.predict(x)
    class_labels = ["Caterpillar", "Diabrotica speciosa", "Healthy"]
    return class_labels[np.argmax(classes)]

#@tool(args_schema=PImage)
def predict_cotton_leaf_disease(img):
    """ Tell whether the cotton leaf has a disease or is healthy """
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    classes = reconstructed_model_cotton_leaf.predict(x)
    class_labels = ["Bacterial blight", "Curl Virus", "Fussarium Wilt", "Healthy"]
    return class_labels[np.argmax(classes)]


#@tool(args_schema=PImage)
def predict_corn_leaf_disease(img):
    """ Tell whether the corn leaf has a disease or is healthy """
    print("Predicting corn leaf disease", img, type(img))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    classes = reconstructed_model_corn_leaf.predict(x)
    class_labels = ["Blight", "Common Rust", "Gray Leaf Spot","Healthy"]
    return class_labels[np.argmax(classes)]

#@tool(args_schema=PImage)
def predict_insect(img):
    """ Find out the insect in the image """
    print("Predicting insect", img, type(img))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    classes = reconstructed_model_insect.predict(x)
    class_labels = ["Ant", "Bee", "Beetle", "Caterpillar", "Earthworm", "Earwig", "Grasshopper", "Moth", "Slug", "Snail", "Wasp", "Weevil"]
    return class_labels[np.argmax(classes)]


@tool(args_schema=Location)
def get_weather_data(latitude, longitude):
    """
    Get the weather data for a given location latitude and longitude.
    """
    latlong = latitude+","+longitude
    url = f"http://api.weatherapi.com/v1/forecast.json?key={weather_api_key}&q={latlong}&days=7"
    response = requests.get(url)
    return response.json()


@tool
def calculate_water_needed(field_moisture, desired_moisture, rainfall_expected, field_area):
    """
    Calculate the amount of water needed to reach the desired moisture level.

    Args:
        field_moisture (float): Current moisture level of the field (0-100%)
        desired_moisture (float): Desired moisture level of the field (0-100%)
        rainfall_expected (float): Expected rainfall in the next 24 hours (inches)
        field_area (float): Area of the field (acres)

    Returns:
        float: Amount of water needed to reach the desired moisture level (gallons)
    """

    # Convert rainfall from inches to gallons per acre
    rainfall_gallons = rainfall_expected * 27154 * field_area

    # Calculate the amount of water needed to reach the desired moisture level
    water_needed = ((desired_moisture - field_moisture) / 100) * field_area * 27154 - rainfall_gallons

    return water_needed


@tool
def increase_ph(current_ph, desired_ph, soil_area_acres):
    """
    Calculate the amount of lime needed to increase soil pH.

    Args:
        current_ph (float): Current soil pH
        desired_ph (float): Desired soil pH
        soil_area_acres (float): Area of the soil in acres

    Returns:
        float: Amount of lime needed (tons)
    """
    ph_increase = desired_ph - current_ph
    lime_needed = ph_increase * soil_area_acres * 4840 / 1000 * 40
    return lime_needed


@tool
def decrease_ph(current_ph, desired_ph, soil_area_acres):
    """
    Calculate the amount of aluminum sulfate needed to decrease soil pH.

    Args:
        current_ph (float): Current soil pH
        desired_ph (float): Desired soil pH
        soil_area_acres (float): Area of the soil in acres
        soil_weight_tons_per_acre (float): Weight of the soil in tons per acre (typically 2 million pounds or 1000 tons per acre)
    Returns:
        float: Amount of aluminum sulfate needed (pounds)
    """
    ph_decrease = current_ph - desired_ph
    aluminum_sulfate_needed = ph_decrease * soil_area_acres * 43560 / 10 * 2
    return aluminum_sulfate_needed


@tool(args_schema=CropQuestion)
def get_crop_info(crop_question, crop):
    """
    Ask a question about the crop that the farmer is growing.
    """
    return retrieval_graph.invoke(crop_question, crop)


@tool(args_schema=CropQuestion)
def fertilizer_to_add(crop_question, crop):
    """
    Get the recommended fertilizer for a specific crop.
    """
    return retrieval_graph.invoke(crop_question, crop)


class CropDisease(BaseModel):
    crop: str = Field(..., description="Crop to protect")
    disease_name: str = Field(..., description="Name of the disease")
    moisture: float = Field(..., description="Soil moisture level")
    weather: str = Field(..., description="Weather forecast")
    irrigation_plan: str = Field(..., description="Irrigation plan recommendation")


@tool(args_schema=CropDisease)
def tackle_disease(crop, disease_name, moisture, weather, irrigation_plan):
    """Get insights on how to address disease for a given crop"""

    prompt_template = PromptTemplate.from_template(
        """
        You are an agricultural disease management expert is a professional with specialized knowledge in entomology, 
        plant pathology, and crop protection.
        
        A farmer has come to you with a disease effeecting his/her crop. 
        The farmer is growing {crop}. 
        The farmer has noticed {disease} disease on the crop.
        His farm's current and next few days weather is {weather}.
        His farm's soil moisture is {moisture}. And his irrigation plan is {irrigation_plan}. 
        
        You need to provide the farmer with the following information:
        1. Insights on the disease, how it effects the plant and its yield
        2. What causes the disease and how to prevent it
        3. Now that the disease is present, how to remediate it? Include specific informaiton
            - On what pesticides to use, when to apply given the weather, moisture and irrigation plan
                - explain your reasoning for the timing. Provide reference to the weather and moisture levels and you used it in your reasoning
                - give dates when the pesticides should be applied
            - Where to get the pesticides from
        """
    )
    question = prompt_template.format(crop=crop, disease=disease_name, moisture=moisture, weather=weather, irrigation_plan=irrigation_plan)

    print("Tackling disease", question, crop)
    return retrieval_graph.invoke(question, crop)


class CropInsect(BaseModel):
    crop: str = Field(..., description="Crop to protect")
    insect_name: str = Field(..., description="Name of the disease")
    moisture: float = Field(..., description="Soil moisture level")
    weather: str = Field(..., description="Weather forecast")
    irrigation_plan: str = Field(..., description="Irrigation plan recommendation")


@tool(args_schema=CropInsect)
def tackle_insect(crop, insect_name, moisture, weather, irrigation_plan):
    """Get insights on how to address insect for a given crop"""

    prompt_template = PromptTemplate.from_template(
        """
        You are an agricultural pest management expert is a professional with specialized knowledge in entomology, 
        plant pathology, and crop protection.

        A farmer has come to you with a disease effeecting his/her crop. 
        The farmer is growing {crop}. 
        The farmer has noticed {insect_name} insect on the crop.
        His farm's current and next few days weather is {weather}.
        His farm's soil moisture is {moisture}. And his irrigation plan is {irrigation_plan}. 

        You need to provide the farmer with the following information:
        1. Insights on the insect, how it effects the plant and its yield
        2. What factors support insect habitation in your crop field
        3. Now that the insects are present, how to remediate it? Include specific informaiton
            - On what pesticides to use, when to apply given the weather, moisture and irrigation plan
                - explain your reasoning for the timing. Provide reference to the weather and moisture levels and you used it in your reasoning
                - give dates when the pesticides should be applied
            - Where to get the pesticides from
                - Give the websites where the farmer can buy the pesticides
        """
    )
    question = prompt_template.format(crop=crop, insect_name=insect_name, moisture=moisture, weather=weather,
                                      irrigation_plan=irrigation_plan)

    print("Tackling insect", question, crop)
    return retrieval_graph.invoke(question, crop)