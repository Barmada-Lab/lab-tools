from datetime import datetime

def parse_datetime(date_str: str, time_str: str) -> datetime:
    MM, DD, YYYY = map(int, date_str.split("/"))
    hh, mm = map(int, time_str.split(" ")[0].split(":"))
    am = time_str.split(" ")[1].lower() == "m"
    if am and hh == 12:
        hh == 0
    elif not am and hh != 12:
        hh += 12
    return datetime(YYYY, MM, DD, hh, mm)
