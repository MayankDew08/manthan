from typing import List
from grader import GradeResult

def find_grade_quality_msg(messages: list, quality: int) -> List[GradeResult]:
    return [msg for msg in messages if msg.quality == quality]
