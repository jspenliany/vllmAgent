import re
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

def parseLog(filename: str):
    pair_count = 0
    with open(filename, "r") as f:
        log_list = f.readlines()
        for logLine in log_list:
            pattern = r"HumanMessage\(content='(.*?)'.*?AIMessage\(content='(.*?)'"

            # re.DOTALL allows the '.' to match newline characters
            pairs = re.findall(pattern, logLine, re.DOTALL)

            if pairs:
                pair_count += 1
                latest_human, latest_ai = pairs[-1]
                # if "雨不感到厌" in latest_human:
                #     pair_count += 1
                #     print(f"{pair_count}--- Latest Pair ---\nHuman: {latest_human}\nAI: {latest_ai}")
                # return latest_human, latest_ai
                print(f"{pair_count}--- Latest Pair ---\nHuman: {latest_human}\nAI: {latest_ai}")

if __name__ == "__main__":
    logFilePath = os.path.join(parentdir, "agent_workflow.log")
    print(logFilePath)
    parseLog(logFilePath)