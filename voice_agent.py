"""
LiveKit Voice Agent with Intelligent Interruption Handling
==========================================================

This is the main agent implementation that integrates the
interrupt handler with LiveKit's voice pipeline.

Author: Kushagra
Assignment: SalesCode.ai GenAI Engineer Campus Recruitment - NSUT
"""

import os
import logging
import asyncio
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
from livekit.plugins import silero, deepgram, openai, cartesia

from interrupt_handler import InterruptHandler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the interrupt handler globally
interrupt_handler = InterruptHandler(
    confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.6")),
    enable_contextual_analysis=True,
    enable_multi_language=True,
    log_events=True
)

@function_tool
async def get_time(context: RunContext):
    """Returns the current time."""
    from datetime import datetime
    current_time = datetime.now().strftime("%H:%M:%S")
    logger.info(f"Time requested: {current_time}")
    return {"time": current_time}

async def entrypoint(ctx: JobContext):
    """
    Main entry point for the LiveKit agent with interruption handling.
    """
    logger.info("="*70)
    logger.info("Starting Voice Agent")
    logger.info("="*70)
    
    await ctx.connect()
    logger.info(f" Connected to room: {ctx.room.name}")
    
    # Load custom filler words
    custom_fillers_en = os.getenv("FILLER_WORDS_EN", "").split(",")
    custom_fillers_hi = os.getenv("FILLER_WORDS_HI", "").split(",")
    if custom_fillers_en and custom_fillers_en[0]:
        interrupt_handler.add_custom_fillers("en", custom_fillers_en)
    if custom_fillers_hi and custom_fillers_hi[0]:
        interrupt_handler.add_custom_fillers("hi", custom_fillers_hi)
    
    # Build agent object
    agent = Agent(
        instructions="""You are an intelligent, friendly voice assistant built with LiveKit.

You are conversational, helpful, and patient. You understand that users may say filler
words like "uh", "umm", or "hmm" while thinking, and you don't get interrupted by these.

However, if a user clearly wants to interrupt you with words like "wait", "stop", or 
"hold on", you immediately stop speaking and listen.

Be natural and engaging in your responses. Keep responses concise unless asked for detail.""",
    )
    
    logger.info("Agent created with interrupt handling")
    
    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.5,
            min_silence_duration=0.5,
            prefix_padding_duration=0.2,
        ),
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
    )
    
    @session.on("agent_speech_started")
    def on_agent_speech_started():
        interrupt_handler.set_agent_speaking(True)
        logger.debug("Agent started speaking")
    
    @session.on("agent_speech_ended")
    def on_agent_speech_ended():
        interrupt_handler.set_agent_speaking(False)
        logger.debug("Agent stopped speaking")
    
    # Process speech through handler - by synchronous callback
    @session.on("user_speech_committed")
    def on_user_speech(transcript: str, confidence: float = 0.9):
        async def process_speech():
            result = await interrupt_handler.process_speech_event(
                transcript=transcript,
                confidence=confidence,
                language="en"
            )
            action = result["action"]
            if action == "IGNORE":
                logger.info(f"Filler detected (ignore): '{transcript}'")
            elif action == "INTERRUPT":
                logger.info(f"Valid interruption: '{transcript}'")
            elif action == "REGISTER":
                logger.info(f"Registering speech: '{transcript}'")
        asyncio.create_task(process_speech())
    
    # Start session
    await session.start(agent=agent, room=ctx.room)
    logger.info("Agent session started and listening")
    
    # Send initial greeting
    await session.generate_reply(
        instructions="Greet the user warmly. Tell them you're an intelligent assistant that understands natural conversation, including when they say 'umm' or 'uh'."
    )
    
    logger.info("Initial greeting sent")
    logger.info("="*70)
    logger.info("Agent is now active and handling interruptions")
    logger.info("="*70)
    
    # Logging stats
    try:
        while True:
            await asyncio.sleep(60)
            stats = interrupt_handler.get_statistics()
            logger.info(f"Stats: {stats['total_events']} events, "
                        f"{stats['ignored_fillers']} fillers ignored, "
                        f"{stats['valid_interrupts']} interrupts processed")
    except asyncio.CancelledError:
        logger.info("Agent session ending...")
        interrupt_handler.print_summary()
        interrupt_handler.export_event_log("interruption_log.json")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
