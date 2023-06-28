import pandas as pd
import streamlit as st
import openai
import os
import utils
import json
import time

def main():
    try:
        st.set_page_config(layout="wide")
        # st.title("FNOL Audio Transcription, Summarization, Entity Extraction and Personalized Email Generation Demo")
        st.markdown("<h1 style='font-size:35px;'>FNOL Audio Transcription, Summarization, Entity Extraction, Personalized Email Generation and Post Call Analysis Demo</h1>", unsafe_allow_html=True)
        st.sidebar.image("logo.png",width = 50)
        st.sidebar.title("Upload FNOL Audio Recording Below")
        # Step 1: Upload .wav audio file
        uploaded_file = st.sidebar.file_uploader("Upload .wav audio file", type=["wav"])

        if uploaded_file:
            audio_data = uploaded_file.read()
            st.sidebar.title("What would you like to do?")
            # Using object notation
            add_selectbox = st.sidebar.selectbox(
                "Choose one option from below",
                ("None","Transcribe FNOL Call Recording", "Generate FNOL Summary", "Extract Key Entities","Generate Personalized Email",
                 "Post Call Analysis")
            )

            if add_selectbox == "Transcribe FNOL Call Recording":
                # Step 2: Transcribe audio using Azure Speech to Text
                placeholder = st.empty()  # Create an empty placeholder
                placeholder.subheader("Transcribing Audio...")
                if uploaded_file.name in ["audio_transcript_1.wav","audio_transcript_2.wav","audio_transcript_3.wav"]:
                    files_dict = {
                        "audio_transcript_1.wav": "audio_transcript_1-transcript.json",
                        "audio_transcript_2.wav": "audio_transcript_2-transcript.json",
                        "audio_transcript_3.wav": "audio_transcript_3-transcript.json"
                    }

                    f = open("transcripts/"+files_dict[uploaded_file.name])
                    transcript_json = json.load(f)
                    try:
                        result = transcript_json["Transcript"]
                    except Exception as e:
                        result = transcript_json["transcript"]
                    time.sleep(10)
                else:
                    result, transcript_json = utils.transcribe_audio(audio_data)
                    result = ",".join(result)
                if result:
                    placeholder.empty()  # Clear the placeholder
                    st.subheader("FNOL Transcript from Audio")
                    st.write(result)

                    st.session_state.result = result
                    st.session_state.transcript_json = transcript_json


            elif add_selectbox == "Generate FNOL Summary":
                # Step 3: Summarize text using Azure OpenAI
                placeholder = st.empty()  # Create an empty placeholder
                placeholder.subheader("Summarizing FNOL Transcript...")
                result = st.session_state.get("result")  # Get the result from session state
                if result:
                    placeholder.empty()  # Clear the placeholder
                    summary = utils.summarize_text(result)
                    # Step 4: Display summarization result
                    def display_list_as_bullet_points(my_list):
                        for item in my_list:
                            st.write(f"- {item}")

                    st.subheader("Summarized Text")
                    display_list_as_bullet_points(summary)
                    # st.write(summary)
                else:
                    st.warning("Transcribe audio first")


            elif add_selectbox == "Extract Key Entities":

                placeholder = st.empty()  # Create an empty placeholder
                placeholder.subheader("Extracting Key Entities...")
                transcript_json = st.session_state.get("transcript_json")  # Get the result from session state
                if transcript_json:
                    placeholder.empty()  # Clear the placeholder
                    entities_df = utils.get_entities(transcript_json)
                    # Step 4: Display Extraction result

                    st.subheader("Extracted Entities")
                    # st.dataframe(entities_df)
                    st.table(entities_df)
                else:
                    st.warning("Transcribe audio first")

            elif add_selectbox == "Generate Personalized Email":

                placeholder = st.empty()  # Create an empty placeholder
                placeholder.subheader("Generating Personalized Email...")
                transcript_json = st.session_state.get("transcript_json")  # Get the result from session state
                if transcript_json:
                    placeholder.empty()  # Clear the placeholder
                    email = utils.generate_email_template(transcript_json)
                    st.subheader("Post FNOL Conversation Personalized Email Response")
                    st.write(email)
                else:
                    st.warning("Transcribe audio first")

            elif add_selectbox == "Post Call Analysis":

                placeholder = st.empty()  # Create an empty placeholder
                placeholder.subheader("Generating Post Call Analysis...")
                transcript_json = st.session_state.get("transcript_json")  # Get the result from session state

                # Loading the transcript -testing
                #
                # f = open('audio_transcript_1-transcript.json')
                # transcript_json = json.load(f)

                if transcript_json:

                    placeholder.empty()  # Clear the placeholder

                    customer_df, agent_df = utils.prepareCustomerAgentDF(transcript_json)
                    st.subheader("Post FNOL Conversation CSR Performance Analytics")
                    # Using columns
                    tab1, tab2 = st.tabs(["Dialogue Analysis", "CSR Performance Metrics"])

                    with tab1:
                        st.markdown("## Customer")
                        col1, col2 ,col3 = st.columns([4, 2, 1])
                        # col1, col2 = st.columns(2)
                        # Display the charts in separate columns
                        with col1:
                            st.markdown("###### Dialogue wise emotion scores")
                            fig_cust = utils.create_plot(customer_df)
                            st.plotly_chart(fig_cust.to_dict(), use_container_width=True)

                        with col2:
                            st.markdown("###### Emotion Distribution (%)")
                            h_fig_cust = utils.create_hist_emotions_plot(customer_df)
                            st.plotly_chart(h_fig_cust.to_dict(), use_container_width=True)

                        with col3:
                            st.markdown("###### Sentiment Distribution %")
                            cus_sent_df = utils.sentiment_perc(customer_df)
                            for idx, row in cus_sent_df.iterrows():
                                st.metric(row.Sentiment, row['%'])
                            # c_pie_fig = utils.sentiment_pie_chart(customer_df,"Sentiment Distribution-Customer")
                            # st.plotly_chart(c_pie_fig.to_dict(), use_container_width=True)

                        st.markdown("## Agent")
                        col4, col5 ,col6 = st.columns([4, 2, 1])
                        # col3, col4 = st.columns(3)
                        with col4:
                            st.markdown("###### Dialogue wise emotion scores")
                            fig = utils.create_plot(agent_df)
                            st.plotly_chart(fig.to_dict(), use_container_width=True)
                        with col5:
                            st.markdown("###### Emotion Distribution (%)")
                            h_fig_agent = utils.create_hist_emotions_plot(agent_df)
                            st.plotly_chart(h_fig_agent.to_dict(), use_container_width=True)

                        with col6:
                            st.markdown("###### Sentiment Distribution %")
                            ag_sent_df = utils.sentiment_perc(agent_df)
                            for idx, row in ag_sent_df.iterrows():
                                st.metric(row.Sentiment, row['%'])
                            # a_pie_fig = utils.sentiment_pie_chart(agent_df,"Sentiment Distribution-Agent")
                            # st.plotly_chart(a_pie_fig.to_dict(), use_container_width=True)
                    with tab2:
                        key_metrics = utils.get_key_metrics(transcript_json)
                        # st.write(key_metrics)
                        for question, answer in key_metrics.items():
                            if not(question == "AI generated CSAT Score (on a scale of 1-5)"):
                                st.write(f"**{question}**")
                                st.write(answer)
                                st.write("\n")
                            else:
                                stars = utils.print_stars(answer)
                                st.write(f"**{question}**")
                                st.write(stars)
                                # st.write(answer)
                                st.write("\n")



                else:
                    st.warning("Transcribe audio first")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()