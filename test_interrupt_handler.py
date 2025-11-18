"""
Comprehensive Test Suite for Interrupt Handler

This test file validates all scenarios required.

Run with: python3 test_interrupt_handler.py
"""

import asyncio
import sys
from interrupt_handler import InterruptHandler


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add_result(self, name, expected, actual, passed):
        self.tests.append({
            "name": name,
            "expected": expected,
            "actual": actual,
            "passed": passed
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        print("\n" + "-"*70)
        print("TEST SUITE SUMMARY")
        print("-"*70)
        for test in self.tests:
            status = "PASS" if test["passed"] else "FAIL"
            print(f"{status} | {test['name']}")
            print(f"      Expected: {test['expected']}, Got: {test['actual']}")
        print("-"*70)
        print(f"Total: {len(self.tests)} tests")
        print(f"Passed: {self.passed} ({self.passed/len(self.tests)*100:.1f}%)")
        print(f"Failed: {self.failed}")
        print("-"*70 + "\n")
        return self.failed == 0


async def run_tests():
    """Run all test scenarios"""
    results = TestResults()
    
    print("\n" + "-"*70)
    print("INTERRUPT HANDLER TESTS - ")
    print("-"*70 + "\n")
    
    handler = InterruptHandler(
        confidence_threshold=0.6,
        enable_contextual_analysis=True,
        enable_multi_language=True,
        log_events=True  # Disable logging for cleaner test output
    )
    
    #1: User filler while agent speaks
    print("Test 1: User filler while agent speaks")
    print("   Input: 'umm' (confidence: 0.85)")
    handler.set_agent_speaking(True)
    result = await handler.process_speech_event("umm", 0.85, "en")
    expected = "IGNORE"
    actual = result["action"]
    passed = (actual == expected)
    results.add_result("Filler while agent speaking", expected, actual, passed)
    print(f"   Result: {actual} {'Passed' if passed else 'Not passed'}")
    print(f"   Reason: {result['reason']}\n")
    
    # 2: User real interruption
    print("Test 2: User real interruption while agent speaks")
    print("   Input: 'wait one second' (confidence: 0.92)")
    handler.set_agent_speaking(True)
    result = await handler.process_speech_event("wait one second", 0.92, "en")
    expected = "INTERRUPT"
    actual = result["action"]
    passed = (actual == expected)
    results.add_result("Priority command interruption", expected, actual, passed)
    print(f"   Result: {actual} {'Passed' if passed else 'Not passed'}")
    print(f"   Reason: {result['reason']}\n")
    
    # 3: User filler while agent quiet
    print("Test 3: User filler while agent quiet")
    print("   Input: 'umm' (confidence: 0.88)")
    handler.set_agent_speaking(False)
    result = await handler.process_speech_event("umm", 0.88, "en")
    expected = "REGISTER"
    actual = result["action"]
    passed = (actual == expected)
    results.add_result("Filler while agent quiet", expected, actual, passed)
    print(f"   Result: {actual} {'Passed' if passed else 'Not passed'}")
    print(f"   Reason: {result['reason']}\n")
    
    # 4: Mixed filler and command 
    print("Test 4: Mixed filler and command")
    print("   Input: 'umm okay stop' (confidence: 0.87)")
    handler.set_agent_speaking(True)
    result = await handler.process_speech_event("umm okay stop", 0.87, "en")
    expected = "INTERRUPT"
    actual = result["action"]
    passed = (actual == expected)
    results.add_result("Mixed filler + command", expected, actual, passed)
    print(f"   Result: {actual} {'Passed' if passed else 'Not passed'}")
    print(f"   Reason: {result['reason']}\n")
    
    # 5: Background noise (low confidence)
    print("Test 5: Background noise (low confidence)")
    print("   Input: 'hmm yeah' (confidence: 0.35)")
    handler.set_agent_speaking(True)
    result = await handler.process_speech_event("hmm yeah", 0.35, "en")
    expected = "IGNORE"
    actual = result["action"]
    passed = (actual == expected)
    results.add_result("Low confidence background noise", expected, actual, passed)
    print(f"   Result: {actual} {'Passed' if passed else 'Not passed'}")
    print(f"   Reason: {result['reason']}\n")
    
    # 6: Hindi filler
    print("Test 6: Hindi filler while agent speaks")
    print("   Input: 'haan' (confidence: 0.88)")
    handler.set_agent_speaking(True)
    result = await handler.process_speech_event("haan", 0.88, "hi")
    expected = "IGNORE"
    actual = result["action"]
    passed = (actual == expected)
    results.add_result("Hindi filler detection", expected, actual, passed)
    print(f"   Result: {actual} {'Passed' if passed else 'Not passed'}")
    print(f"   Reason: {result['reason']}\n")
    
    # 7: Dynamic filler addition
    print("Test 7: Dynamic filler addition (runtime config)")
    print("   Adding custom filler 'basically' for English")
    handler.add_custom_fillers("en", ["basically"])
    handler.set_agent_speaking(True)
    result = await handler.process_speech_event("basically", 0.85, "en")
    expected = "IGNORE"
    actual = result["action"]
    passed = (actual == expected)
    results.add_result("Dynamic filler addition", expected, actual, passed)
    print(f"   Result: {actual} {'Passed' if passed else 'Not passed'}")
    print(f"   Reason: {result['reason']}\n")
    
    # 8: Priority Hindi command
    print("Test 8: Priority Hindi command")
    print("   Input: 'ruko' (Hindi for stop) (confidence: 0.90)")
    handler.set_agent_speaking(True)
    result = await handler.process_speech_event("ruko", 0.90, "hi")
    expected = "INTERRUPT"
    actual = result["action"]
    passed = (actual == expected)
    results.add_result("Hindi priority command", expected, actual, passed)
    print(f"   Result: {actual} {'Passed' if passed else 'Not passed'}")
    print(f"   Reason: {result['reason']}\n")
    
    # 9: Empty/whitespace input
    print("Test 9: Edge case - Empty input")
    print("   Input: '' (confidence: 0.70)")
    handler.set_agent_speaking(True)
    result = await handler.process_speech_event("", 0.70, "en")
    expected = "IGNORE"
    actual = result["action"]
    passed = (actual == expected)
    results.add_result("Empty input handling", expected, actual, passed)
    print(f"   Result: {actual} {'Passed' if passed else 'Not passed'}")
    print(f"   Reason: {result['reason']}\n")
    
    # 10: Meaningful speech while agent quiet
    print("Test 10: Meaningful speech while agent quiet")
    print("   Input: 'what is the weather' (confidence: 0.95)")
    handler.set_agent_speaking(False)
    result = await handler.process_speech_event("what is the weather", 0.95, "en")
    expected = "REGISTER"
    actual = result["action"]
    passed = (actual == expected)
    results.add_result("Meaningful speech while quiet", expected, actual, passed)
    print(f"   Result: {actual} {'Passed' if passed else 'Not passed'}")
    print(f"   Reason: {result['reason']}\n")
    
    # 11: Multiple fillers 
    print("Test 11: Multiple fillers in sequence")
    print("   Input: 'uh umm hmm' (confidence: 0.85)")
    handler.set_agent_speaking(True)
    result = await handler.process_speech_event("uh umm hmm", 0.85, "en")
    expected = "IGNORE"
    actual = result["action"]
    passed = (actual == expected)
    results.add_result("Multiple fillers", expected, actual, passed)
    print(f"   Result: {actual} {'Passed' if passed else 'Not passed'}")
    print(f"   Reason: {result['reason']}\n")
    
    #12: Question while agent speaking
    print("Test 12: User asks question while agent speaking")
    print("   Input: 'how does that work' (confidence: 0.92)")
    handler.set_agent_speaking(True)
    result = await handler.process_speech_event("how does that work", 0.92, "en")
    expected = "INTERRUPT"
    actual = result["action"]
    passed = (actual == expected)
    results.add_result("Question interruption", expected, actual, passed)
    print(f"   Result: {actual} {'Passed' if passed else 'Not passed'}")
    print(f"   Reason: {result['reason']}\n")
    
    # Print final statistics
    handler.print_summary()
    
    # Print test results summary
    all_passed = results.print_summary()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    """Run the test suite"""
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)
