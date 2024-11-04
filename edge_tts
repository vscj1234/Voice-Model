import os
import signal
import asyncio
import speech_recognition as sr
from openai import OpenAI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import edge_tts

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Function to read text from .txt files in the /data directory
def read_data_files():
    texts = []
    data_dir = 'data'  # Make sure this directory exists
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                    texts.append(file.read())
    return texts

# Maximum number of tokens for the response
MAX_TOKENS = 50

class Agent:
    def __init__(self, llm):
        self.llm = llm
        self.recognizer = sr.Recognizer()
        self.additional_knowledge = read_data_files()

    def handle(self, user_input, chat_history):
        raise NotImplementedError("Agent needs to implement a handle method.")

    def speech_to_text(self, timeout=5) -> str | None:
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)

                text = self.recognizer.recognize_google(audio)
                print(f"Recognized text: {text}")
                return text
        except sr.WaitTimeoutError:
            print("Listening timeout")
            return None
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    async def text_to_speech(self, text):
        communicate = edge_tts.Communicate(text)
        await communicate.save("response.mp3")
        os.system("start response.mp3")  # This will play the audio on Windows

class S2SAgent(Agent):
    def __init__(self, llm, appointment_agent):
        super().__init__(llm)
        self.appointment_agent = appointment_agent
        self.chat_history = []

    def handle(self, user_input):
        self.chat_history.append({"role": "user", "content": user_input})
        
        # Check if the user input is related to appointment booking
        if self.is_appointment_related(user_input):
            response = self.appointment_agent.handle(user_input, self.chat_history)
        else:
            try:
                # Include additional knowledge in the prompt
                knowledge_prompt = "\n".join(self.additional_knowledge)
                full_prompt = f"{knowledge_prompt}\n\nUser: {user_input}"

                system_message = (
                    "You are a helpful assistant. "
                    f"Provide complete responses within a strict {MAX_TOKENS} token limit. "
                    "Ensure your response is coherent and ends naturally, even if brief."
                )
                
                response = self.llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=0.7
                )
                response = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error generating response: {e}")
                response = "I apologize, but I encountered an error generating a response."

        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def is_appointment_related(self, user_input):
        appointment_keywords = ["appointment", "schedule", "book", "slot", "available", "time"]
        return any(keyword in user_input.lower() for keyword in appointment_keywords)

class AppointmentAgent(Agent):
    def __init__(self, llm):
        super().__init__(llm)

    def handle(self, user_input, chat_history):
        # Use the LLM to handle the appointment conversation dynamically
        try:
            # Include any additional knowledge as context, if necessary
            knowledge_prompt = "\n".join(self.additional_knowledge)
            full_prompt = f"{knowledge_prompt}\n\nUser: {user_input}\n\nChat History: {chat_history}"

            system_message = (
                "You are an appointment booking assistant. "
                f"Provide complete responses within a strict {MAX_TOKENS} token limit. "
                "Ensure your response is coherent and ends naturally, even if brief."
            )

            # Prompt the LLM with the user input and chat history
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.7
            )
            # Extract the response from the LLM
            response_content = response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating appointment response: {e}")
            response_content = "I apologize, but I encountered an error while trying to handle the appointment booking."

        return response_content


# Initialize OpenAI client
client = OpenAI(api_key='sk-cloudjune-')

# Initialize agents
appointment_agent = AppointmentAgent(client)
agent = S2SAgent(client, appointment_agent)

# Global flag for controlling the listening loop
listening = False

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

def signal_handler(signum, frame):
    print("Received signal to terminate")
    global listening
    listening = False
    os._exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global listening
    
    try:
        while True:
            message = await websocket.receive_text()
            
            if message == "start":
                listening = True
                await websocket.send_text("status:Listening started")
                
                while listening:
                    text = agent.speech_to_text(timeout=5)
                    
                    if text:
                        response = agent.handle(text)
                        print(f"AI Response: {response}")
                        await agent.text_to_speech(response)
                        await websocket.send_text(f"response:{response}")
                    
                    # Check if a new message has been received
                    try:
                        new_message = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                        if new_message == "stop":
                            listening = False
                            await websocket.send_text("status:Listening stopped")
                            print("Listening stopped")
                            break
                    except asyncio.TimeoutError:
                        pass
                    
            elif message == "stop":
                listening = False
                await websocket.send_text("status:Listening stopped")
                print("Listening stopped")
            
            else:
                # Handle text-based input
                response = agent.handle(message)
                print(f"AI Response (Text): {response}")
                await websocket.send_text(f"response:{response}")
                
    except WebSocketDisconnect:
        listening = False
        manager.disconnect(websocket)
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error in WebSocket connection: {e}")
        listening = False
        manager.disconnect(websocket)

@app.get("/")
async def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
