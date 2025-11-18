"""
Voice Interruption Handler for LiveKit Agents
Filler detection with fuzzy matching for as natural as possible conversation.

"""

import re
import logging
import asyncio
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import json

logger = logging.getLogger(__name__)

@dataclass
class InterruptionEvent:
    timestamp: datetime
    transcript: str
    confidence: float
    was_agent_speaking: bool
    decision: str  # "IGNORE", "INTERRUPT", "REGISTER"
    reason: str
    language: Optional[str] = "en"
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "transcript": self.transcript,
            "confidence": self.confidence,
            "was_agent_speaking": self.was_agent_speaking,
            "decision": self.decision,
            "reason": self.reason,
            "language": self.language
        }

class MultiLanguageFillerDetector:
    """
    Handles any variation of filler sounds (umm, ummm, ummmm, etc.)
    """
    
    def __init__(self):
        self.filler_words: Dict[str, Set[str]] = {
            "en": {"uh", "um", "umm", "hmm", "hm", "ah", "er", "erm", "like", "you know", "yeah"},
            "hi": {"haan", "hmm", "acha", "theek", "haan ji", "arrey", "toh"},
            "es": {"eh", "este", "pues", "bueno", "mm"},
            "fr": {"euh", "beh", "bon", "hein"},
        }
        
        # Priority interruption words
        self.priority_words: Set[str] = {
            "wait", "stop", "no", "hold", "pause", "cancel", "hang", "one", "second",
            "ruk", "ruko", "nahi", "bas"  # Hindi equivalents
        }
        
        # Patterns that indicate real speech vs fillers
        self.meaningful_patterns = [
            r'\b(what|how|when|where|why|who)\b',
            r'\b(can|could|would|should|will|do|does|did)\b',
            r'\b(please|sorry|excuse|thanks|thank)\b',
            r'\b(yes|okay|ok|sure|alright)\b',
        ]
        
    def add_filler_words(self, language: str, words: List[str]):
        """Dynamically add filler words for a language"""
        if language not in self.filler_words:
            self.filler_words[language] = set()
        self.filler_words[language].update(word.strip().lower() for word in words if word.strip())
        logger.info(f"Added {len(words)} filler words for language '{language}'")
    
    def is_filler_sound(self, word: str, fillers: Set[str]) -> bool:
        """
        Matches any repetition/extension of core filler patterns.
        
        Examples:
        - "um" matches "um", "umm", "ummm", "ummmm", "ummmmm..."
        - "hm" matches "hm", "hmm", "hmmm", "hmmmm..."
        - "uh" matches "uh", "uhh", "uhhh", "uhhhh..."
        - "haan" matches "haan", "haaan", "haaaan..."
        """
        word = word.lower().strip()
        
        # Check each known filler
        for filler in fillers:
            filler_len = len(filler)
            
            # For very short fillers (1-2 chars), be very aggressive
            if filler_len <= 2:
                # Extract unique characters from filler
                filler_chars = set(filler)
                
                # Check if word contains only those characters (repeated any number of times)
                word_chars = set(word)
                if word_chars.issubset(filler_chars) and len(word) >= filler_len:
                    # Word is made only of filler characters
                    return True
            
            # For 3-char fillers like "umm", "hmm"
            elif filler_len == 3:
                # Check if word starts with first char and contains mainly the repeated char
                if len(word) >= 2:
                    first_char = filler[0]
                    repeated_char = filler[1]  # Usually the repeated one like 'm' in "umm"
                    
                    if word[0] == first_char:
                        # Count how many of the repeated char appear
                        repeated_count = word.count(repeated_char)
                        if repeated_count >= 2 and repeated_count / len(word) >= 0.6:
                            return True
            
            # For longer fillers (4+ chars), check for character repetition
            else:
                # Extract core pattern (first few chars)
                core = filler[:2]
                if word.startswith(core):
                    # Check if remaining chars are repetitions from filler
                    filler_chars = set(filler)
                    word_chars = set(word)
                    if word_chars.issubset(filler_chars):
                        return True
        
        return False
    
    def is_filler_only(self, text: str, language: str = "en") -> bool:
        """
        Determines if text contains only filler words/sounds.
        This should return True ONLY for filler sounds,
        not for meaningful words.
        Returns True if text is purely filler, False if meaningful content.
        """
        if not text or not text.strip():
            return True

        normalized_text = text.lower().strip()
        
        # Priority commands are always meaningful
        text_words = normalized_text.split()
        if any(word in self.priority_words for word in text_words):
            return False
        
        # Check for meaningful patterns
        for pattern in self.meaningful_patterns:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                return False
        
        # Split into words
        words = re.findall(r'\b\w+\b', normalized_text)
        if not words:
            return True
        
        # Get all fillers for this language
        language_fillers = self.filler_words.get(language, set())
        english_fillers = self.filler_words.get("en", set())
        all_fillers = language_fillers | english_fillers
        
        # Count fillers with aggressive fuzzy matching
        filler_count = 0
        for word in words:
            # Exact match
            if word in all_fillers:
                filler_count += 1
            # Fuzzy sound match
            elif self.is_filler_sound(word, all_fillers):
                filler_count += 1
        
        # For single-word inputs, must be 100% filler
        # For multi-word - 80% threshold
        threshold = 1.0 if len(words) == 1 else 0.8
        filler_ratio = filler_count / len(words)
        
        return filler_ratio >= threshold
    
    def contains_priority_command(self, text: str) -> bool:
        """Check if text contains priority interruption command"""
        normalized_text = text.lower().strip()
        words = normalized_text.split()
        return any(word in self.priority_words for word in words)
    
    def get_all_fillers(self) -> Set[str]:
        """Get all filler words across all languages"""
        all_fillers = set()
        for fillers in self.filler_words.values():
            all_fillers.update(fillers)
        return all_fillers

class ContextualBufferAnalyzer:
    
    def __init__(self, buffer_duration_ms: int = 150):
        self.buffer_duration_ms = buffer_duration_ms
        self.speech_buffer: deque = deque(maxlen=10)
        
    def add_event(self, transcript: str, confidence: float, timestamp: datetime):
        self.speech_buffer.append({
            "transcript": transcript,
            "confidence": confidence,
            "timestamp": timestamp
        })
    
    def get_context(self) -> Dict:
        if not self.speech_buffer:
            return {
                "avg_confidence": 0.0,
                "confidence_trend": 0.0,
                "word_count_trend": 0.0,
                "is_escalating": False,
                "min_confidence": 0.0,
                "max_confidence": 0.0
            }
        
        events = list(self.speech_buffer)
        confidences = [e["confidence"] for e in events]
        word_counts = [len(e["transcript"].split()) for e in events]
        
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        confidence_trend = 0.0
        word_count_trend = 0.0
        
        if len(events) >= 2:
            confidence_trend = confidences[-1] - confidences[0]
            word_count_trend = word_counts[-1] - word_counts[0]
        
        is_escalating = confidence_trend > 0.1 and word_count_trend > 0
        
        return {
            "avg_confidence": avg_confidence,
            "confidence_trend": confidence_trend,
            "word_count_trend": word_count_trend,
            "is_escalating": is_escalating,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence
        }

class InterruptHandler:
    """
    KEY BEHAVIOR:
    - When agent IS speaking: Ignore fillers, allow real interruptions
    - When agent NOT speaking: Treat everything as valid speech
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        enable_contextual_analysis: bool = True,
        enable_multi_language: bool = True,
        log_events: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.enable_contextual_analysis = enable_contextual_analysis
        self.enable_multi_language = enable_multi_language
        self.log_events = log_events
        
        # Initialize components
        self.filler_detector = MultiLanguageFillerDetector()
        self.context_analyzer = ContextualBufferAnalyzer() if enable_contextual_analysis else None
        
        # Agent state
        self.is_agent_speaking = False
        
        # Statistics
        self.event_log: List[InterruptionEvent] = []
        self.stats = {
            "total_events": 0,
            "ignored_fillers": 0,
            "valid_interrupts": 0,
            "registered_speech": 0,
            "low_confidence_ignored": 0,
            "fuzzy_filler_matches": 0,
            "avg_confidence": 0.0,
        }
        
        logger.info(f" InterruptHandler initialized:")
        logger.info(f"  - Confidence threshold: {confidence_threshold}")
        logger.info(f"  - Contextual analysis: {enable_contextual_analysis}")
        logger.info(f"  - Multi-language: {enable_multi_language}")
        logger.info(f"  - Aggressive fuzzy filler matching: ENABLED")
    
    def set_agent_speaking(self, is_speaking: bool):
        """Update agent's speaking state"""
        self.is_agent_speaking = is_speaking
        logger.debug(f"Agent speaking state: {is_speaking}")
    
    def add_custom_fillers(self, language: str, fillers: List[str]):
        """Dynamically add filler words at runtime"""
        self.filler_detector.add_filler_words(language, fillers)
        logger.info(f"Added {len(fillers)} custom fillers for '{language}'")
    
    async def process_speech_event(
        self,
        transcript: str,
        confidence: float,
        language: str = "en"
    ) -> Dict[str, any]:
        """
        CORE LOGIC:
        1. Low confidence → IGNORE (noise)
        2. Priority command → INTERRUPT (always)
        3. Agent speaking + filler → IGNORE
        4. Agent speaking + real speech → INTERRUPT
        5. Agent quiet + anything → REGISTER (treat as valid)
        """
        timestamp = datetime.now()
        self.stats["total_events"] += 1
        
        # Update confidence stats
        total_conf = self.stats["avg_confidence"] * (self.stats["total_events"] - 1)
        self.stats["avg_confidence"] = (total_conf + confidence) / self.stats["total_events"]
        
        # Context buffering
        if self.context_analyzer:
            self.context_analyzer.add_event(transcript, confidence, timestamp)
            context = self.context_analyzer.get_context()
        else:
            context = {}
        
        # Make decision
        decision, reason = await self._make_decision(
            transcript, confidence, language, context
        )
        
        # Log event
        event = InterruptionEvent(
            timestamp=timestamp,
            transcript=transcript,
            confidence=confidence,
            was_agent_speaking=self.is_agent_speaking,
            decision=decision,
            reason=reason,
            language=language
        )
        
        if self.log_events:
            self.event_log.append(event)
            logger.info(f"[{decision}] '{transcript}' | Conf: {confidence:.2f} | {reason}")
        
        # Update stats
        if decision == "IGNORE":
            self.stats["ignored_fillers"] += 1
            if "fuzzy" in reason.lower():
                self.stats["fuzzy_filler_matches"] += 1
        elif decision == "INTERRUPT":
            self.stats["valid_interrupts"] += 1
        elif decision == "REGISTER":
            self.stats["registered_speech"] += 1
        
        return {
            "action": decision,
            "reason": reason,
            "should_stop_agent": (decision == "INTERRUPT"),
            "metadata": {
                "confidence": confidence,
                "was_agent_speaking": self.is_agent_speaking,
                "language": language,
                "context": context
            }
        }
    
    async def _make_decision(
        self,
        transcript: str,
        confidence: float,
        language: str,
        context: Dict
    ) -> tuple:
        # Rule 1: Low confidence → Ignore (background noise)
        if confidence < self.confidence_threshold:
            self.stats["low_confidence_ignored"] += 1
            return "IGNORE", f"low_confidence ({confidence:.2f} < {self.confidence_threshold})"
        
        # Rule 2: Priority commands → ALWAYS interrupt
        if self.filler_detector.contains_priority_command(transcript):
            return "INTERRUPT", "priority_command_detected"
        
        # Rule 3: Check if filler-only (with aggressive fuzzy matching)
        is_filler_only = self.filler_detector.is_filler_only(transcript, language)
        
        # Rule 4: Agent speaking + filler → IGNORE
        if self.is_agent_speaking and is_filler_only:
            return "IGNORE", "filler_while_agent_speaking"
        
        # Rule 5: Agent speaking + meaningful → INTERRUPT
        if self.is_agent_speaking and not is_filler_only:
            return "INTERRUPT", "meaningful_speech_while_agent_speaking"
        
        # Rule 6: Agent NOT speaking → REGISTER (all speech is valid)
        if not self.is_agent_speaking:
            if is_filler_only:
                return "REGISTER", "filler_while_agent_quiet"
            else:
                return "REGISTER", "speech_while_agent_quiet"
        
        # Default
        return "REGISTER", "default_registration"
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        return {
            **self.stats,
            "ignore_rate": self.stats["ignored_fillers"] / max(self.stats["total_events"], 1),
            "interrupt_rate": self.stats["valid_interrupts"] / max(self.stats["total_events"], 1),
            "fuzzy_match_rate": self.stats["fuzzy_filler_matches"] / max(self.stats["ignored_fillers"], 1) if self.stats["ignored_fillers"] > 0 else 0
        }
    
    def export_event_log(self, filepath: str = "interruption_log.json"):
        """Export event log to JSON"""
        with open(filepath, 'w') as f:
            json.dump([event.to_dict() for event in self.event_log], f, indent=2)
        logger.info(f"Exported {len(self.event_log)} events to {filepath}")
    
    def print_summary(self):
        """Print performance summary"""
        stats = self.get_statistics()
        print("\n" + "-"*60)
        print("INTERRUPT HANDLER - SUMMARY")
        print("-"*60)
        print(f"Total events processed: {stats['total_events']}")
        print(f"Ignored fillers: {stats['ignored_fillers']} ({stats['ignore_rate']:.1%})")
        print(f"  └─ Fuzzy matches: {stats['fuzzy_filler_matches']}")
        print(f"Valid interrupts: {stats['valid_interrupts']} ({stats['interrupt_rate']:.1%})")
        print(f"Registered speech: {stats['registered_speech']}")
        print(f"Low confidence ignored: {stats['low_confidence_ignored']}")
        print(f"Average confidence: {stats['avg_confidence']:.2f}")
        print("-"*60 + "\n")

if __name__ == "__main__":
    import asyncio
    
    async def test_handler():
        """Test fuzzy matching"""
        handler = InterruptHandler(
            confidence_threshold=0.6,
            enable_contextual_analysis=True,
            enable_multi_language=True
        )
        
        handler.set_agent_speaking(True)
        
        test_cases = [
            ("umm", 0.85, "en"),
            ("ummm", 0.85, "en"),
            ("ummmm", 0.85, "en"),
            ("ummmmm", 0.85, "en"),
            ("hmmmmmmmmmmmmmmmmm", 0.8, "en"),
            ("hmm", 0.85, "en"),
            ("hmmm", 0.85, "en"),
            ("hmmmm", 0.85, "en"),
            ("haan", 0.88, "hi"),
            ("haaan", 0.88, "hi"),
            ("haaaan", 0.88, "hi"),
            ("wait stop", 0.92, "en"),
            ("stop", 0.4, "en"),
            ("hello", 0.7, "en"),
        ]
        
        print("\n" + "-"*60)
        print("Testing fuzzy filler matching -")
        print("-"*60 + "\n")
        
        for transcript, confidence, language in test_cases:
            result = await handler.process_speech_event(transcript, confidence, language)
            status = "Filler or noise" if result["action"] == "IGNORE" else "Stopped"
            print(f"{status} '{transcript}' → {result['action']}")
        
        handler.print_summary()
    
    asyncio.run(test_handler())
