import streamlit as st
import requests
import json # Import json for better error handling/display

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Multi-Model Router",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Content ---
with st.sidebar:
    st.header("Router Capabilities")
    st.markdown("""
    This application demonstrates an intelligent routing system. 
    Your prompt is sent to a backend model (e.g., an Ollama classification model) which determines the required capability:
    
    - **IMAGE_GENERATION**: Creates static visuals (e.g., "Draw a cat in the rain").
    - **VIDEO_GENERATION**: Creates dynamic content (e.g., "Generate a short clip of a sunset").
    - **AUDIO_GENERATION**: Converts text to speech or handles audio (e.g., "Convert this text to an audio file").
    - **TRANSLATION**: Translates text between languages (e.g., "Translate English to Marathi").
    - **TEXT_GENERATION**: Handles all standard text tasks (e.g., "Write a poem," "Summarize a topic").
    """)


# --- Main Application Layout ---
st.title("🤖 Intelligent Model Router")
st.caption("Automatically route your requests to the correct AI model.")

# --- Input Area ---
prompt = st.text_area(
    "💡 Enter your prompt here:",
    height=120,
    placeholder="E.g. Generate a 5-second animation of a car driving through a city.",
    key="user_prompt"
)

# Use a container for the output to keep it clean
output_container = st.container()

if st.button("🚀 Generate", use_container_width=True, type="primary"):
    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt before generating.", icon="🚨")
    else:
        # Use st.status for a better, interactive loading state
        with st.status("Routing request and generating output...", expanded=True) as status:
            try:
                # 1. Classification & Routing Request
                status.update(label="1/2: Sending prompt to router backend...", state="running")
                
                # Mock endpoint as per the user's requirement
                router_url = "http://localhost:5400/generate"
                
                response = requests.post(router_url, json={"prompt": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    action = data.get("action", "TEXT_GENERATION").upper()
                    augmented_prompt = data.get("augmented_prompt", "N/A")
                    result = data.get("result", "No result returned.")

                    # 2. Output Presentation
                    status.update(label=f"2/2: Capability detected: **{action}**", state="running")

                    with output_container:
                        st.subheader(f"✅ Capability: {action}")
                        
                        # Display augmented prompt and metadata in an expander
                        with st.expander("🔍 Prompt Details & Metadata"):
                            st.markdown("**Detected Capability:**")
                            st.code(action)
                            st.markdown("**Augmented Prompt Sent to Generator:**")
                            st.code(augmented_prompt, language='text')

                        # Handle different capabilities
                        st.markdown("---")
                        
                        if action == "IMAGE_GENERATION":
                            st.markdown("### 🖼️ Generated Image")
                            # Assuming 'result' is a publicly accessible URL for the image
                            st.image(result, caption=augmented_prompt[:50] + "...", use_container_width=True)

                        elif action == "VIDEO_GENERATION":
                            st.markdown("### 🎬 Generated Video")
                            # Assuming 'result' is a publicly accessible URL for the video
                            st.video(result)

                        elif action == "AUDIO_GENERATION":
                            st.markdown("### 🎧 Generated Audio")
                            # Assuming 'result' is a publicly accessible URL for the audio file
                            st.audio(result)
                            
                        elif action == "TRANSLATION":
                            st.markdown("### 🌍 Translation Result")
                            st.success(f"**Translated Output:**\n\n{result}")

                        elif action == "TEXT_GENERATION":
                            st.markdown("### 📝 Text Generation Output")
                            st.info(result)
                            
                        else:
                            st.markdown("### ❓ Unknown Output Type")
                            st.write(result)
                            
                    status.update(label="✅ Generation Complete!", state="complete")
                    
                else:
                    # Handle non-200 responses
                    error_text = response.text
                    try:
                        error_json = response.json()
                        error_text = json.dumps(error_json, indent=2)
                    except:
                        pass
                    
                    st.error(f"❌ Router Error ({response.status_code}):", icon="🔥")
                    st.code(error_text)
                    status.update(label=f"Request failed with status {response.status_code}", state="error")

            except requests.exceptions.ConnectionError:
                st.error("❌ Connection Error: Could not connect to the backend server at `http://localhost:5400/generate`. Please ensure your server is running.", icon="🔌")
                status.update(label="Connection Failed", state="error")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}", icon="💥")
                status.update(label="An internal error occurred", state="error")
