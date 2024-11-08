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
import tempfile
from playsound import playsound

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create a temporary directory for audio files
temp_dir = tempfile.mkdtemp()

def read_data_files():
    texts = []
    data_dir = 'data'
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                    texts.append(file.read())
    return texts

MAX_TOKENS = 50

class Agent:
    def __init__(self, llm):
        self.llm = llm
        self.recognizer = sr.Recognizer()
        self.additional_knowledge = read_data_files()
        self.current_audio_file = None

    def handle(self, user_input, chat_history):
        raise NotImplementedError("Agent needs to implement a handle method.")

    def speech_to_text(self, timeout=15) -> str | None:
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

    def text_to_speech(self, text):
        try:
            # Clean up previous audio file if it exists
            if self.current_audio_file and os.path.exists(self.current_audio_file):
                os.remove(self.current_audio_file)

            # Create new audio file with timestamp to avoid conflicts
            self.current_audio_file = os.path.join(temp_dir, f'response_{id(text)}.mp3')
            
            # Generate speech using OpenAI's TTS API
            response = self.llm.audio.speech.create(
                model="tts-1",
                voice="nova",  # Use 'nova' for a natural female voice
                input=text
            )
            
            # Save the audio file
            response.stream_to_file(self.current_audio_file)
            
            # Play the audio
            playsound(self.current_audio_file)
            
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    def cleanup(self):
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            try:
                os.remove(self.current_audio_file)
            except Exception as e:
                print(f"Error cleaning up audio file: {e}")

    def create_system_prompt(self) -> str:
        return """You are CloudJune's AI assistant. Follow these rules strictly:
1. For general greetings and personal questions:
   - Respond naturally and warmly to greetings
   - Introduce yourself as CJ, CloudJune's AI assistant
   - Be friendly and conversational while maintaining professionalism
   - For personal questions, provide engaging responses while being clear about your role as an AI
2. ONLY answer questions using information from the provided context
3. If the question cannot be answered using the provided context, respond with: "I apologize, but I can't find information about that in my knowledge base. Could you please ask something related to CloudJune's services?"
4. Never make up or infer information not present in the context
5. Keep responses concise and direct
6. Include specific references to where in the knowledge base you found the information
7. Interaction Style:
   - Be engaging and friendly while remaining professional
   - Use natural language and conversational tone
   - Show empathy and understanding in responses
   - Be helpful and proactive in guiding users
8. For appointment-related queries, defer to the appointment system
Current context will be provided in the user message."""

class S2SAgent(Agent):
    def __init__(self, llm, appointment_agent):
        super().__init__(llm)
        self.appointment_agent = appointment_agent
        self.chat_history = []

    def handle(self, user_input):
        self.chat_history.append({"role": "user", "content": user_input})
        
        if self.is_appointment_related(user_input):
            response = self.appointment_agent.handle(user_input, self.chat_history)
        else:
            try:
                knowledge_prompt = "\n".join(self.additional_knowledge)
                full_prompt = f"{self.create_system_prompt()}\n\n{knowledge_prompt}\n\nUser: {user_input}"
                
                response = self.llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant with additional knowledge."},
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=MAX_TOKENS
                )
                response = response.choices[0].message.content
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
        try:
            knowledge_prompt = "\n".join(self.additional_knowledge)
            full_prompt = f"{self.create_system_prompt()}\n\n{knowledge_prompt}\n\nUser: {user_input}\n\nChat History: {chat_history}"

            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",  # Updated to use gpt-4
                messages=[
                    {"role": "system", "content": "You are an appointment booking assistant."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=MAX_TOKENS
            )
            response_content = response.choices[0].message.content

        except Exception as e:
            print(f"Error generating appointment response: {e}")
            response_content = "I apologize, but I encountered an error while trying to handle the appointment booking."

        return response_content

# Initialize OpenAI client (rest of the code remains the same)
client = OpenAI(api_key='sk-proj-****')  # Replace with your actual API key

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
    # Cleanup
    agent.cleanup()
    # Clean up temp directory
    try:
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Error cleaning up temp directory: {e}")
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
            command = await websocket.receive_text()
            
            if command == "start":
                listening = True
                await websocket.send_text("status:Listening started")
                
                while listening:
                    text = agent.speech_to_text(timeout=5)
                    
                    if text:
                        response = agent.handle(text)
                        print(f"AI Response: {response}")
                        
                        agent.text_to_speech(response)
                        
                        await websocket.send_text(f"response:{response}")
                    
                    await asyncio.sleep(0.1)
                    
            elif command == "stop":
                listening = False
                agent.cleanup()
                await websocket.send_text("status:Listening stopped")
                
    except WebSocketDisconnect:
        listening = False
        agent.cleanup()
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error in WebSocket connection: {e}")
        listening = False
        agent.cleanup()
        manager.disconnect(websocket)

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    agent.cleanup()
    # Clean up temp directory
    try:
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Error cleaning up temp directory: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # Run the application with: uvicorn app3:app --reload
