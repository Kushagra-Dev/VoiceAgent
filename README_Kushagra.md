# README.md – Intelligent Interruption Handling for LiveKit Agent

**Author:** Kushagra Sabharwal
**GitHub Link:** https://github.com/Kushagra-Dev/VoiceAgent/tree/feature/livekit-interrupt-handler-Kushagra


## What Changed

- Added `intelligent_interrupt_handler.py` to the agent pipeline for interruption filtering.
- Integrated interrupt handler with LiveKit callbacks (`user_speech_committed`, `agent_speech_started`, `agent_speech_ended`).
- Configurable filler word lists per language; supports runtime updates via environment variables or code.
- Aggressive fuzzy matching detects extended filler patterns (e.g., "umm", "ummm", "ummmmm").
- Logic adapts decisions based on agent speaking state, speech confidence, and priority commands.
- Real-time statistics and event logging added for development and debugging.


## What Works

- Agent ignores fillers ("umm", "haan", etc.) while speaking, preventing unwanted interruptions.
- Valid speech and priority commands (e.g., "wait", "stop", "ruko") immediately interrupt the agent.
- Fillers are properly registered as valid input when the agent is quiet.
- All main logic verified by running `test_interrupt_handler.py` with 12/12 tests passing, including edge cases for confidence, multilingual input, and custom fillers.
- Dynamic configuration tested for both English and Hindi filler lists.


## Known Issues

- LiveKit's upstream VAD/TTS pipeline sometimes pauses agent prematurely before the interrupt handler receives the transcript. This is a platform-level limitation.
- When LLM or TTS API quota is exhausted (e.g., OpenAI or Cartesia), agent replies may fail to generate.Balance must be added for longer sessions.
- Some filler patterns or edge cases with very low confidence/noise may occasionally be missed or classified incorrectly.


## Steps to Test

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Add API keys to `.env` (or environment):
    ```
    OPENAI_API_KEY=sk-xxxxxxx
    CARTESIA_API_KEY=sk-yyyyyyy
    ```

3. Configure custom filler lists:
    ```
    FILLER_WORDS_EN=uh,umm,hmm,like
    FILLER_WORDS_HI=haan,acha,theek
    ```

4. Start the agent in console mode:
    ```
    python3 livekit_agent_with_interrupts.py console
    ```

5. Manual test:
    - While the agent speaks, say various filler sounds ("umm", "hmm") — agent should not be interrupted.
    - Say a real command ("wait", "stop", "one second") — agent should interrupt immediately.
    - Speak fillers when agent is quiet — agent should register turn and process input.

6. Automated test:
    ```
    python3 test_interrupt_handler.py
    ```
    - All expected cases should show PASS.

## Environment Details

- Python version: 3.13.5 (works with 3.9+)
- Main dependencies:
    - livekit-agents >= 1.3.0
    - python-dotenv
    - openai
    - deepgram
    - cartesia
- Configuration parameters:
    - CONFIDENCE_THRESHOLD=0.6 (set default, can be changed)
    - Custom filler words and language codes (FILLER_WORDS_EN, FILLER_WORDS_HI)
