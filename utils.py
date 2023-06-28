import openai
import azure.cognitiveservices.speech as speechsdk
import streamlit as st
# from azure.identity import DefaultAzureCredential
# from azure.keyvault.secrets import SecretClient
import time
import yaml
import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import requests
import ast # for dictionary extraction
import plotly.express as px
import re

with open("config.yaml", "r") as c:
    config = yaml.safe_load(c)

with open("secrets.yaml", "r") as s:
    secrets = yaml.safe_load(s)


# def get_key(vault_name, key_name, key_version):
#     # Create DefaultAzureCredential
#     credential = DefaultAzureCredential()

#     # Create SecretClient
#     vault_url = f"https://{vault_name}.vault.azure.net/"
#     client = SecretClient(vault_url=vault_url, credential=credential)

#     # Fetch the OpenAI API key from Azure Key Vault
#     api_key_secret = client.get_secret(key_name, key_version)
#     api_key = api_key_secret.value

#     return api_key


SPEECH_TO_TEXT_ENDPOINT = config["SPEECH_TO_TEXT_ENDPOINT"]
SPEECH_TO_TEXT_KEY = "da7b88a45437478481a75bc64038d3a7"
# SPEECH_TO_TEXT_KEY = get_key(vault_name= secrets["VAULT_NAME"], 
#                                    key_name= secrets["STT_KEY_NAME"], 
#                                    key_version= secrets['STT_KEY_VERSION'])


OPENAI_API_KEY = "2e74a7804d404204b9700b088a1e3eaf"
# OPENAI_API_KEY = get_key(vault_name= secrets["VAULT_NAME"], 
#                                    key_name= secrets["OPENAI_KEY_NAME"], 
#                                    key_version= secrets['OPENAI_KEY_VERSION'])

openai.api_type = config["API_TYPE"]
openai.api_version = config["API_VERSION"]
openai.api_base = config["OPENAI_API_BASE"]
openai.api_key = OPENAI_API_KEY

# with open('prompts.json') as file:
#     # Load the JSON data
#     prompts = json.load(file)


def transcribe_audio(audio_data):

    # Check if the result is already cached
    if "transcription" in st.session_state and 'transcript_json' in st.session_state:
        return st.session_state.transcription,st.session_state.transcript_json

    import tempfile
    # create a list to hold the recognized text
    all_results = []

    # create a temporary file to write the audio data
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(audio_data)
        temp_audio_name = temp_audio.name

    speech_config = speechsdk.SpeechConfig(endpoint= SPEECH_TO_TEXT_ENDPOINT, subscription= SPEECH_TO_TEXT_KEY)
    audio_config = speechsdk.audio.AudioConfig(filename=temp_audio_name)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # define a function to handle the final recognition result
    def handle_final_result(evt):
        all_results.append(evt.result.text)

    # define a function to stop continuous recognition
    done = False
    def stop_cb(evt):
        speech_recognizer.stop_continuous_recognition()
        nonlocal done
        done = True

    # register the handle_final_result and stop_cb functions to the SpeechRecognizer events
    speech_recognizer.recognized.connect(handle_final_result)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # start continuous recognition
    speech_recognizer.start_continuous_recognition()

    # wait for recognition to finish
    while not done:
        time.sleep(.5)

    # For entity extraction
    transcript_json = {}

    transcript_json["transcript"] = ".".join(all_results)
    # delete the local audio file
#     os.remove(local_file_name)
    # Store the result in the session state for caching
    st.session_state.transcription = all_results
    st.session_state.transcript_json = transcript_json


    return all_results,transcript_json


def summarize_text(text):

    if "summarization" in st.session_state:
        return st.session_state.summarization
    
    prompt = f"""Your task is to summarise the below text between 350 to 400 words and in 4 paragrpahs./
                 Extract the policy number, customer name, agent name from the transcript and include in the summary./
                 Also convert the text which looks like phone number and convert to numerical phone number and include in the summary./
                 Focus on how the accident happened and the damages occured while generating the summary./
                 Mention if there is any injury for either the customer or third party./
                 Also include in the summary if there was any emergency services like ambulance, police, tow services, paramedic, fire services etc. was called on the scene."""

    prompt += "\n\ntext: " + "".join(text)

    response = openai.Completion.create(
        engine = config["GPT_DEPLOYMENT_NAME"],
        prompt = prompt,
        max_tokens = config["SUMMARISATION_MAX_TOKENS"],
        n = config["SUMMARISATION_N"],
        stop = None,
        temperature = config["SUMMARISATION_TEMPERATURE"]
    )

    # summary = response.choices[0].text.split("\n")
    summary = response.choices[0].text.strip()
    summary = summary.split("\n")

    summary = [x for x in summary if x]

    # summary = ["• " + sentence for sentence in summary]
    # save temporarily
    temp_str = "\n".join(summary) + "\n\n"
    # temp_dict = {}
    # temp_dict["summary"] = temp_str
    # with open("summary.json", "w") as file:
    #     json.dump(temp_dict, file)

    # Store the result in the session state for caching
    st.session_state.summarization = summary
    return summary


def get_response(prompt):
    response = openai.Completion.create(
        engine= config["GPT_DEPLOYMENT_NAME"],
        prompt= prompt,
        max_tokens= config["ENT_EXT_MAX_TOKEN"],
        temperature= config["ENT_EXT_TEMPERATURE"]
    )

    res_list = response.choices[0].text.split("\n")

    return res_list


def get_entities(transcript_json):
    # f = open('transcripts/audio_transcript_3-transcript.json')
    # json_file = json.load(f)
    try:

        if "extraction" in st.session_state:
            return st.session_state.extraction

        json_file = transcript_json

        try:
            transcript = json_file["Transcript"]
        except Exception:
            transcript = json_file["transcript"]


        #Prompt generation for entity extraction
        prompt1 = f"""
            Extract the following accident information of the customer provided in the transcript delimited by triple backticks. If no information is provided then return "not available"
            If date of accident is not in date format then return "Not available"
            -Number of parties or vehicles involved in the accident./
            -Did the customer get injured ?/
            -Extract the injury details of the customer./
            -Did anyone else get injured other than the customer?/
            -Extract the injury details of third party./
            -Who is the reporting party?/
            -Date of accident./
            -Time of accident./
            -What was location of accident?/
            
            transcript: '''{transcript}'''
            """

        #Prompt generation for entity extraction
        prompt2 = f"""
        
        Extract the following vehicle information of the customer provided in the transcript delimited by triple backticks. If no information is provided then return "Not available"
        VIN and Registration number or Plate number cannot be same
        
        -VIN of the customer vehicle./
        -Registration number or Plate Number of the vehicle./
        -Make of the vehicle./
        -Model of the vehicle./
        -Year of Manufacture of the vehicle./
        -Describe damages to the customer vehicle in minute details.
        -Registration number or Plate Number of the third party vehicle./
        -Make of the third party vehicle./
        -Model of the third party vehicle./
        -Describe damages to the third party vehicle in minute details.
        
        transcript: '''{transcript}'''
        """

        #Prompt generation for entity extraction
        prompt3 = f"""
        
        
        Extract the following accident related information provided in the transcript delimited by triple backticks. If the information is not provided then return "Not available"
        -Direction of travel of the customer vehicle. Return "Stationary vehicle" if the vehicle wasn't moving/
        -Direction of travel of the third party vehicle. Return "Stationary vehicle" if the vehicle wasn't moving/
        -Evasive actions taken by the customer/
        -Evasive actions taken by the third party
        
        transcript: '''{transcript}'''
        """

        #Prompt generation for entity extraction
        prompt4 = f"""
        
        Based the following weather related information provided in the transcript delimited by triple backticks. 
        Answer each of the below as a boolean value.If the information is not provided then return "Not available".
        
        -Light rain/
        -heavy rain/
        -Snow/
        -Fog/
        -Sunny/
        -Cloudy
        
        transcript: '''{transcript}'''
        """

        #Prompt generation for entity extraction
        prompt5 = f"""
        Extract the lighting conditions of the accident scene return boolean result for the below lightning conditions .Return "Not available" if no information is available
        - dark/
        - well lit/
        - moderate lightning/
        
        transcript: '''{transcript}'''
        """

        #Prompt generation for entity extraction
        prompt6 = f"""
        Extract the flowing information from the transcript delimited by triple backticks. If information is not present return "not available"
        
        - Where was the customer travelling to with his vehicle
        
        - cause of the loss
          
        transcript: '''{transcript}'''
        """

        #Prompt generation for entity extraction
        prompt7 = f"""
        
        Extract the below information from the transcript provided in triple backticks.Return "not available" if information is not present.
        
        -Parts of the customer vehicle impacted by the accident. Return an estimated severity score between 1 to 10 for each damaged vehicle part where 1 is the lowest and 10 is the highest 
        
        -Vehicle's current location while talking to the agent
        
        -injury information of customer. Detail the physical injury if any and hospitalisation details
        
        -injury information of other parties. Detail the physical injury if any and hospitalisation details
        
        transcript: '''{transcript}'''
        """

        #Prompt generation for entity extraction
        prompt8 = f"""
        
        Extract information regarding the below emergency services from the transcript enclosed in triple backicks and return a boolean output.  Return "Not available" if no information is provided
        
        -Fire services
        -Police
        -Ambulance
        -Paramedic
        -Animal Services
        -Forest services
        
        transcript: '''{transcript}'''
        """

        #Prompt generation for entity extraction
        prompt9 = f"""
        
        Extract information regarding the below additional services from the transcript enclosed in triple backicks.  Return "Not available" if no information is provided
        -Tow service availed
        -Did the customer arrange the towing service by himself
        -Name of the towing service used
        -Rental services availed
        -Did the customer arrange the rental services by himself
        -First Aid provided at the accident scene
        -Repair Needed
        -Options provided to customer for repair shop/body shop
        -Choice of repair/body shop by the customer
        -Road side assistance
        
        transcript: '''{transcript}'''
        """

        #Prompt generation for entity extraction
        prompt10 = f"""
        
        Extract information regarding the road condition at the time of accident from the transcript enclosed in triple backicks and return boolean value for each of them.  Return "Not available" if no information is provided
        -Wet Road
        -Potholes
        -Road underconstruction
        -Low traffic
        -Heavy traffic
        -One way road
        -Defective road sign
        
        transcript: '''{transcript}'''
        """

        #Prompt generation for entity extraction
        prompt11 = f"""
        
        Extract information from the transcript enclosed in triple backicks.  Return "Not available" if no information is provided
        -Damages description to the surrounding area or state/public property
        -Fatality details in the accident. Return a boolean value
        -Fatality details description
        -Drivability of the vehicle
        -Rental coverage on the policy
        -Details about witnesses of the accident
        -Receipts or invoices availability related to accident
        -Pictures or video availability of damaged vehicle
        -Police report number
        -Hospitalisation details
        
        transcript: '''{transcript}'''
        """

        pmpts = [prompt1,prompt2,prompt3,prompt4,prompt5,prompt6,prompt7,prompt8,prompt9,prompt10,prompt11]

        final_result = []
        for prompt in pmpts:
            final_result.extend(get_response(prompt))

        # Cleaning out empty strings at the beginning of each list
        final_result = [x for x in final_result if x]

        # Initialize an empty dictionary
        info_dict = {}

        # Iterate over each element in the list
        for item in final_result:

            try:

                # Split the item by the first occurrence of ":"
                key, value = item.split(":", 1)

                # Remove leading/trailing spaces from the key and value
                key = key.strip()
                value = value.strip()

                # Add key-value pair to the dictionary
                info_dict[key] = value

                # with open("extraction.json", "w") as file:
                #     json.dump(info_dict, file)
            except Exception as e:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(e)

        final_df = pd.DataFrame(info_dict.items(), columns=["Entity","Value"])


        # Store the result in the session state for caching
        st.session_state.extraction = final_df
        return final_df
    except Exception as e:
        print("Error while extracting entities")
        print(e)


def generate_email_template(transcript_json):

    if "email_temp" in st.session_state:
        return st.session_state.email_temp

    json_file = transcript_json

    try:
        transcript = json_file["Transcript"]
    except Exception:
        transcript = json_file["transcript"]


    # defining the prompt

    email_prompt = f"""

Act as an auto email genarator bot. Your task is to generate a personalised email based on the conversation transcript delimited by triple backticks. 
Follow the below instructions while generating the summary

Instructions
    - The email should have 200 words 
    - There should be four sections
        - Subject
        - Greeting
        - Main email body
        - email signature.

        Instruction for email subject
            - Add an arbitrary 6 digit claim request number in the mail subject line

        Instruction for Greeting
            - Greet the customer as "Dear Customer"

        Instruction for main email body
        - Express appreciation for customer's call regarding the accident claim 
        - Add the same claim number as generated in the subject of the email (Refer the instruction for email subject above)
        - Extract damages to the vehicle
        - Explain the actions to be taken by customer and insurance provider
        - Add "Please feel free to reply to this email if you have any further queries or require additional assistance." in the body
        - Don't include any email address that are mentioned in the transcript
        - Don't ask for feedback in the email body

        Instruction for email Signature
            - End the email with a courteous closing and company's name in two different lines

transcript: '''{transcript}'''
"""

    # Getting response

    response = openai.Completion.create(
        engine=config["GPT_DEPLOYMENT_NAME"],
        prompt=email_prompt,
        max_tokens= config["EMAIL_MAX_TOKEN"],
        temperature= config["EMAIL_TEMPERATURE"]
    )

    email = response.choices[0].text
    #temporary save
    # temp_dict = {}
    # temp_dict["email"] = str(email)
    # with open("email.json", "w") as file:
    #     json.dump(temp_dict, file)

    # Store the result in the session state for caching
    st.session_state.email_temp = email

    return email


def extract_dictionary_from_string(string):
    # Find the opening and closing curly braces to identify the dictionary substring
    start_index = string.find("{")
    end_index = string.rfind("}")

    # Extract the dictionary substring from the string
    dictionary_str = string[start_index:end_index+1]

    # Convert the dictionary string into a dictionary object
    dictionary = ast.literal_eval(dictionary_str)

    return dictionary

def chunk_transcript(transcript_json):
    try:
        json_file = transcript_json

        try:
            transcript = json_file["Transcript"]
        except Exception:
            transcript = json_file["transcript"]

        # Chunking transcript based on total length
        # print("----------- Printing transcript ------------")
        # print(transcript)
        input_string = transcript

        # Calculate the length of the string
        string_length = len(input_string)

        # Divide the length of string by 4
        chunk_length = int(string_length / 4)

        print("------ printing chunk length -----------",chunk_length)

        # Split the input string into 4 chunks
        chunks = [input_string[i:i + chunk_length] for i in range(0, string_length, chunk_length)]

        # Removing chunk with less that 10 characters
        chunks = [chunk for chunk in chunks if len(chunk) > 10]
    except Exception as e:
        print("Error while chunking")
        print(e)

    return chunks


def create_prompt(transcript):
    try:

        prompt = f""" Using the transcript enclosed in triple backticks which is a conversation between an insurance agent
        and a customer, perform the below steps to accomplish the task

        1. Based on the context, Identify the dialogues spoken by the customer or the insurance agent.
        2. Label each dialogue as "Customer:" or "Agent:" at the beginning of the sentence
        3. Display the results in the order of the dialogues been spoken.
        4. Recheck the entire transcript to see if all the dialogues are covered before displaying the results.

        transcript: '''{transcript}'''


        """

        # prompt = """
        # Using the transcript enclosed in triple backticks which is a conversation between an insurance agent
        # and a customer, perform the below steps to accomplish the task
        #
        # 1. Based on the context, Identify the dialogues spoken by the customer or the insurance agent.
        # 2. Label each dialogue as "Customer:" or "Agent:" at the beginning of the sentence
        # 3. Display the results in the order of the dialogues been spoken.
        # 4. Recheck the entire transcript to see if all the dialogues are covered before displaying the results.
        #
        # transcript: '''{}'''
        # """.format(transcript)

    except Exception as e:
        print("Error while creating prompt")
        print(e)

    return prompt


def get_response_di(prompt):
    try:
        response = openai.Completion.create(
            engine='iipins-gpt-3',
            prompt=prompt,
            max_tokens=1600,
            temperature=0.1
        )
    except Exception as e:
        print("Error while getting response")
        print(e)
    return response


def dialogueIdentification(transcript):
    try:
        chunks = chunk_transcript(transcript)
        final_result = []
        for chunk in chunks:
            # response = ""
            prompt = create_prompt(chunk)
            response = get_response_di(prompt)
            final_result.append(response.choices[0].text)
        print("------ printing dialogueIdentification -----------", len(final_result))
    except Exception as e:
        print("Error at dialogueIdentification")
        print(e)
    return final_result


def dialoguesSeparation(final_result):
    try:
        # Combining all the chunks
        final_result = ",".join(final_result).split("\n")
        final_result = [ele.lstrip() for ele in final_result]

        customer_list = []
        agent_list = []

        for item in final_result:
            if item.startswith('Customer'):
                customer_list.append(item.replace('Customer:', "").strip())
            elif item.startswith('Agent'):
                agent_list.append(item.replace('Agent:', "").strip())
        print("------ printing dialoguesSeparation -customer-----------", len(customer_list))
        print("------ printing dialoguesSeparation -agent-----------", len(agent_list))
    except Exception as e:
        print("Error at dialoguesSeparation")
        print(e)
    return customer_list, agent_list


def get_sentiment_scores(emotion_list):
    try:
        sid = SentimentIntensityAnalyzer()
        scores = {}
        for emotion in emotion_list:
            sentiment_scores = sid.polarity_scores(emotion)
            scores[emotion] = sentiment_scores['compound']
        return scores
    except Exception as e:
        print("Error at get_sentiment_scores")
        print(e)


def scoringEmotions():
    try:
        emotions_df = pd.DataFrame()
        emotions = ['Appreciative', 'Grateful', 'Understanding', 'Helpful', 'Patience', 'Relieved', 'Compassionate',
                    'Sympathetic', 'Apologetic', 'Upset', 'Frustrated', 'Worried', 'Angry', 'Terrified', 'Rude',
                    'Anxiety', 'Fear', 'Stressed']
        # List of emotions

        emotion_list = emotions

        # Get sentiment scores for emotions
        emotion_scores = get_sentiment_scores(emotion_list)
        # Creating a dataframe to store emotions and its respective score

        emotions_df = pd.DataFrame(emotion_scores.items(), columns=["Emotion", "score"])
    except Exception as e:
        print("Error at scoringEmotions")
        print(e)

    return emotions_df


def get_gpt_response(prompt):
    response = openai.Completion.create(
        engine='iipins-gpt-3',
        prompt=prompt,
        max_tokens=1600,
        temperature=0.1  # or 0.5
    )

    return response


def get_emotions_per_dialogue(dialogue_list):
    prompt = f"""
    For each of the insurance agent dialogues in the list enclosed in triple backticks.
    Identify the suitable emotion from below
    'Appreciative','Grateful','Understanding','Helpful','Patience','Relieved','Compassionate','Sympathetic','Apologetic','Upset','Frustrated','Worried','Angry','Terrified','Rude','Anxiety','Fear','Stressed'
    The output should be a dictionary with sentence as the key and emotion enclosed in quotes as the value
    list:'''{dialogue_list}'''
"""

    return prompt


def prepareCustomerAgentDF(transcript):
    try:
        if "sentiment" in st.session_state:
            return st.session_state.sentiment

        customer_df = pd.DataFrame()
        agent_df = pd.DataFrame()
        final_result_ca = dialogueIdentification(transcript)
        customer_list, agent_list = dialoguesSeparation(final_result_ca)
        # Preparing emotions_df

        emotions_df = scoringEmotions()
        # For Customer

        cust_prompt = get_emotions_per_dialogue(customer_list)

        response = get_gpt_response(cust_prompt)

        # cust_dict = ast.literal_eval(response.choices[0].text)
        cust_dict = extract_dictionary_from_string(response.choices[0].text)

        # For Agent

        agent_prompt = get_emotions_per_dialogue(agent_list)

        response = get_gpt_response(agent_prompt)

        # agent_dict = ast.literal_eval(response.choices[0].text)
        agent_dict = extract_dictionary_from_string(response.choices[0].text)

        print("------ printing after attaching emotions-----------", len(cust_dict),len(agent_dict))


        # Creating dataframes with dialogues and emotions for customer and agent
        customer_df = pd.DataFrame(cust_dict.items(), columns=["Dialogue", "Emotion"])

        agent_df = pd.DataFrame(agent_dict.items(), columns=["Dialogue", "Emotion"])

        #         #assigning scores for each emotion
        customer_df = pd.merge(customer_df, emotions_df, on="Emotion", how="left")
        agent_df = pd.merge(agent_df, emotions_df, on="Emotion", how="left")

        # Calculating sentiment for each dialogue
        customer_df["Sentiment"] = customer_df["score"].apply(
            lambda x: "Positive" if x > 0 else ("Neutral" if x == 0 else "Negative"))
        agent_df["Sentiment"] = agent_df["score"].apply(
            lambda x: "Positive" if x > 0 else ("Neutral" if x == 0 else "Negative"))

        print("---------- Printing final dataframe shape ---------- ",customer_df.shape,agent_df.shape)

        # Saving results for caching
        st.session_state.sentiment = (customer_df, agent_df)
    except Exception as e:
        print("Error at prepareCustomerAgentDF")
        print(e)

    return customer_df, agent_df


def create_plot(df):

    try:
        # Add a new column for dialogue number
        df['Dialogue_num'] = range(1, len(df) + 1)

        print("------------- Printing DF details ------------")
        print(df.shape)
        print(df.columns)

        # Create a new column for line color based on score
        df['Color'] = df['score'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')

        # Define color mapping
        color_map = {'Positive': 'green', 'Negative': 'red'}

        # Create a Plotly figure
        # fig = px.line(df, x='Dialogue_num', y='score', color='Color', color_discrete_map=color_map,
        #               hover_data=['Dialogue', 'Emotion', 'score'])

        fig = px.line(df, x='Dialogue_num', y='score',hover_data=['Dialogue', 'Emotion', 'score'],
                      color_discrete_sequence=['#ffe600'])

        # Add a line at y=0
        fig.add_hline(y=0, line_dash='dash', line_color='red')

        # Customize tooltips
        fig.update_traces(hovertemplate="<b>Dialogue:</b> %{customdata[0]}<br><b>Emotion:</b> %{customdata[1]}<br><b>score:</b> %{y:.2f}")

        # Customize layout
        fig.update_layout(
            xaxis_title='Dialogue Number',
            yaxis_title='Score',
            hovermode='x'
        )

        return fig
    except Exception as e:
        print(" ----- Error at Create plot --------")
        print(e)


def create_hist_emotions_plot(df):
    try:


        # Calculate the count and percentage of emotions
        emotion_count = df['Emotion'].value_counts().sort_index()
        emotion_percentage = round((emotion_count / emotion_count.sum()) * 100,3)

        # Sort the emotions in ascending order of count
        emotion_count_sorted = emotion_count.sort_values()
        emotion_percentage_sorted = emotion_percentage.loc[emotion_count_sorted.index]


        fig = px.bar(y=emotion_count_sorted.index, x=emotion_count_sorted,
                     orientation='h',
                     color_discrete_sequence=['#ffe600'])

        # Update the tooltip to display the percentage
        # fig.update_traces(hovertemplate='Emotion: %{y}<br>Percentage: %{text:.2f}%',
        #                   text=emotion_percentage_sorted)
        fig.update_traces(hovertemplate='Percentage: %{text:.2f}%',
                          text=emotion_percentage_sorted)
        # Remove the text labels on the bars
        fig.update_layout(showlegend=False, xaxis_showticklabels=False)


        return fig
    except Exception as e:
        print(" Error at create_hist_emotions_plot ")
        print(e)


def get_key_metrics(transcript_json):
    try:
        json_file = transcript_json

        try:
            transcript = json_file["Transcript"]
        except Exception:
            transcript = json_file["transcript"]
        km_dict = {}
        prompt = f"""

    Using the transcript enclosed in triple backticks which is a conversation between a customer and an insurance agent to do 
    the following, your task is to answer the below listed questions by giving proper explanation to each one of them .

    - Did the insurance agent properly greet the customer and introduced himself
    - Have the insurance agent put forward the call recording disclosure
    - Did the insurance agent gather customer details and verified his identity
    - Did the insurance agent had a clear understanding of the customer query throughout the conversation
    - How was the query resolution ability of the insurance agent
    - What level of product or service knowledge did the insurance agent have
    - Did the insurance agent attempt for any cross selling opportunities
    - Did the insurance agent showcase any bias/racism/unfairness and discrimination towards the customer
    - Did the insurance agent request for any customer sensitive/confidential information such as OTP,CVV,PIN,SSN,etc
    - Was the insurance agent courteous throughout the conversation
    - Did the insurance agent comply with the pre-defined script
    - Was the insurance agent compliant with the closure script
    - Did the insurance agent request for a feedback from the customer
    - Did the customer request/issue resolved during the connect
    - Can you give CSAT score for this conversation on a scale of 5 
    - Suggest any potential areas of improvement for the insurance agent 

    The output should be a dictionary with key as the question and value as your answer to it.

    transcript:'''{transcript}'''

    """

        response = openai.Completion.create(
            engine='iipins-gpt-3',
            prompt=prompt,
            max_tokens=max(4096 - len(prompt), 1200),
            temperature=0.2
        )

        # try:
        #     if "Answer:\n" in response.choices[0].text:
        #         km_dict = ast.literal_eval(response.choices[0].text.split("Answer:\n")[1].lstrip())
        #     elif "Output:\n" in response.choices[0].text:
        #         km_dict = ast.literal_eval(response.choices[0].text.split("Output:\n")[1].lstrip())
        # except Exception:
        #     km_dict = ast.literal_eval(response.choices[0].text)

        km_dict = extract_dictionary_from_string(response.choices[0].text)

        km_df = pd.DataFrame(km_dict.items(), columns=["Question", "Answer"])

        map_dict = {
            "Did the insurance agent properly greet the customer and introduced himself": "Greetings and Introduction",
            "Have the insurance agent put forward the call recording disclosure": "Call Recording Disclosure",
            "Did the insurance agent gather customer details and verified his identity": "Identity Verification",
            "Did the insurance agent had a clear understanding of the customer query throughout the conversation": "Understanding of Customer Query",
            "How was the query resolution ability of the insurance agent": "Query Resolution Ability",
            "What level of product or service knowledge did the insurance agent have": "Product/ Service Knowledge",
            "Did the insurance agent attempt for any cross selling opportunities": "Leverage Cross Sell Opportunity",
            "Did the insurance agent showcase any bias/racism/unfairness and discrimination towards the customer": "Bias/Racist/Unfair Comments",
            "Did the insurance agent request for any customer sensitive/confidential information such as OTP,CVV,PIN,SSN,etc": "Sensitive & Confidential Financial Information",
            "Was the insurance agent courteous throughout the conversation": "Conversation Courtesy",
            "Did the insurance agent comply with the pre-defined script": "Overall Script Compliance",
            "Was the insurance agent compliant with the closure script": "Closure Script Compliance",
            "Did the insurance agent request for a feedback from the customer": "Request for Feedback",
            "Did the customer request/issue resolved during the connect": "First Call Resolution",
            "Can you give CSAT score for this conversation on a scale of 5": "AI generated CSAT Score (on a scale of 1-5)",
            "Suggest any potential areas of improvement for the insurance agent": "Potential Areas of Improvement"}

        km_df["Question"] = km_df["Question"].map(map_dict)
        result_dict = km_df.set_index('Question')['Answer'].to_dict()

        return result_dict
    except Exception as e:
        print("Error at get_key_metrics")
        print(e)


def print_stars(statement):
    # Extract the CSAT score using regular expressions
    csat_score_match = re.search(r"\d+(\.\d+)?", statement)
    if csat_score_match:
        csat_score = float(csat_score_match.group())
    else:
        csat_score = None

    # Display the number of star emojis based on the CSAT score using Streamlit emojis
    if csat_score is not None:
        num_stars = int(csat_score)
        has_half_star = csat_score - num_stars >= 0.5

        # Handle the case when the CSAT score is equal to the maximum scale
        if num_stars == 5:
            whole_stars = "⭐️" * num_stars
        else:
            # Create a string with whole star emojis
            whole_stars = "⭐️" * num_stars

            # Create a string with half star emoji if applicable
            if has_half_star:
                whole_stars += "½"

        # Display the stars using Streamlit
    #         print(whole_stars)
    else:
        print("No valid CSAT score found in the statement.")

    return whole_stars

def sentiment_pie_chart(df,title):
    # Count the number of occurrences for each sentiment category
    sentiment_counts = df['Sentiment'].value_counts()

    # Define custom colors in hex format
    custom_colors = ['#ffe600', '#cccccc', '#999999']
    # Create a pie chart using Plotly Express
    fig = px.pie(sentiment_counts, values=(sentiment_counts.values/len(df))*100, names=sentiment_counts.index,
                 color_discrete_sequence=custom_colors,title=title)

    return fig

def sentiment_perc(df):
    # Calculate percentage for each sentiment

    percentage_df = pd.DataFrame((df["Sentiment"].value_counts() / len(df)) * 100).reset_index().rename(columns={'count': '%'})
    percentage_df["%"] = round(percentage_df["%"], 2)
    return percentage_df












