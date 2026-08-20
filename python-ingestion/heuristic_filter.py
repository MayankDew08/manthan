"""Remove obvious low-signal chat records before expensive LLM grading."""

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict
from parser import Data


def heuristic_filter(messages: List[Data]) -> Tuple[List[Data], List[Dict]]:
    """Return retained messages plus structured reasons for obvious discards."""
    kept = []
    discarded = []
    
    for msg in messages:
        text = msg.text.strip()
        reason = None
        
        # Rule 1: Completely empty
        if not text:
            reason = "empty"
        
        # Rule 2: Media only, zero caption
        elif msg.is_media and len(text.replace("<Media omitted>", "").strip()) == 0:
            reason = "media_only_no_caption"
        
        # Rule 3: Pure emoji / punctuation (no alphanumeric characters)
        # Catches: "👍", "😂", "🔥", "...", "!!!"
        elif not any(c.isalnum() for c in text):
            reason = "no_alphanumeric"
        
        # Rule 4: Very short pure acknowledgment (< 4 words, no question mark, no URL)
        # Catches: "lol", "haha", "ok", "thanks", "ty", "gg", "nice", "wow"
        # BUT preserves: "lol what?", "ok thanks for the link", "nice article"
        elif (
            len(text.split()) <= 3 
            and "?" not in text 
            and "http" not in text.lower()
            and text.lower() in {
                "lol", "lmao", "haha", "hehe", "ok", "okay", "k", 
                "thanks", "thank you", "ty", "thx", 
                "nice", "cool", "wow", "gg", "yep", "yes", "no", "nah",
                "same", "true", "agreed", "exactly", "right", "correct",
                "got it", "makes sense", "understood", "clear",
                "gm", "gn", "good morning", "good night", "good evening",
            }
        ):
            reason = "short_acknowledgment"
        
        if reason:
            discarded.append({
                "timestamp": msg.datetime_iso,
                "sender": msg.sender,
                "text": text[:200],
                "reason": reason,
                "stage": "heuristic"
            })
        else:
            kept.append(msg)
    
    return kept, discarded
