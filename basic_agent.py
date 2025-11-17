import os
import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import silero, deepgram, openai, elevenlabs

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@function_tool
async def get_weather(
    context: RunContext,
    location: str,
):
    """
    Looks up weather information for a given location.
    This is a mock function - replace with real API call if needed.
    """
    logger.info(f"Weather lookup requested for: {location}")
    return {
        "location": location,
        "weather": "sunny",
        "temperature": 72,
        "humidity": 65,
    }

@function_tool
async def get_time(context: RunContext):
    """Returns the current time."""
    from datetime import datetime
    current_time = datetime.now().strftime("%H:%M:%S")
    logger.info(f"Time requested: {current_time}")
    return {"time": current_time}

async def entrypoint(ctx: JobContext):
    """
    This function sets up and runs the voice agent.
    
    Args:
        ctx: JobContext containing room, user, and other session info
    """
    logger.info("Agent starting...")
    
    await ctx.connect()
    logger.info(f"Connected to room: {ctx.room.name}")

    agent = Agent(
        instructions="""You are a friendly and helpful voice assistant. 
        You are warm, conversational, and eager to help users with their questions.
        You have access to tools to get weather and time information.
        Always be concise and natural in your responses.
        If a user interrupts you, stop speaking immediately and listen to what they have to say.""",
        tools=[get_weather, get_time],
    )
    
    logger.info("Agent created with tools: get_weather, get_time")

    # VAD: Voice Activity Detection - detects when user is speaking
    # STT: Speech-to-Text - converts user voice to text
    # LLM: Language Model - generates intelligent responses
    # TTS: Text-to-Speech - converts responses back to voice
    
    session = AgentSession(
        vad=silero.VAD.load(),                    # Detects speech activity
        stt=deepgram.STT(model="nova-3"),         # Speech recognition
        llm=openai.LLM(model="gpt-4o-mini"),      # Language model (using mini for cost)
        tts=elevenlabs.TTS(),                     # Voice synthesis
    )
    
    logger.info("Agent session created with:")
    logger.info("  - VAD: Silero")
    logger.info("  - STT: Deepgram (nova-3)")
    logger.info("  - LLM: OpenAI (gpt-4o-mini)")
    logger.info("  - TTS: ElevenLabs")

    await session.start(agent=agent, room=ctx.room)
    logger.info("Agent session started")

    await session.generate_reply(
        instructions="Greet the user and ask how you can help them today."
    )
    logger.info("Initial greeting sent")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))