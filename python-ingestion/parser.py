from dataclasses import dataclass
import datetime
import re

@dataclass
class Data:
    datetime_iso : str
    sender : str
    text : str
    is_media : bool

message_re = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?::\d{2})?\s*[APap][Mm])(?:\s*-\s*)?"
)

DATE_FORMATS = ("%d/%m/%y", "%d/%m/%Y")
TIME_FORMATS = ("%I:%M:%S %p", "%I:%M %p")


def parse_datetime(date_str: str, time_str: str) -> datetime.datetime:
    time_str = time_str.replace("\u202f", " ").replace("\u00a0", " ").strip()
    for date_fmt in DATE_FORMATS:
        for time_fmt in TIME_FORMATS:
            try:
                return datetime.datetime.strptime(
                    f"{date_str} {time_str}", f"{date_fmt} {time_fmt}"
                )
            except ValueError:
                continue
    raise ValueError(f"cannot parse datetime: {date_str} {time_str}")


def parse_chat(file_path: str) -> list[Data]:
    messages = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            match = message_re.match(line)
            if match:
                date_str, time_str = match.groups()
                rest = line[match.end():]
                if ": " in rest:
                    sender, text = rest.split(": ", 1)
                    sender = sender.strip()
                else:
                    continue
                messages.append(Data(
                    parse_datetime(date_str, time_str).isoformat(),
                    sender,
                    text,
                    "<Media omitted>" in text,
                ))
            elif messages and line:
                messages[-1].text += "\n" + line
    return messages


if __name__ == "__main__":
    messages = parse_chat("tests/test_chat.txt")
    print(f"total messages: {len(messages)}")
    print("\n--- first 10 ---")
    for msg in messages[:10]:
        print(msg)
    print("\n--- last 10 ---")
    for msg in messages[-10:]:
        print(msg)
