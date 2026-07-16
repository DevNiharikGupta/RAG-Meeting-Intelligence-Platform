import os
import json
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ---- config from .env ----
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "tinyllama")


# ---- The prompt that tells the LLM exactly what to extract ----
EXTRACTION_PROMPT = """You are a meeting note taker. Read the transcript below and extract key information.

Answer in this exact JSON format (replace the example values with real data from the transcript):

Example:
{{
  "action_items": [
    {{"owner": "John", "task": "Write the report", "deadline": "Friday"}}
  ],
  "decisions": [
    {{"decision": "Use Python for the project", "made_by": "John"}}
  ],
  "key_discussion_points": [
    {{"topic": "Project timeline", "summary": "Team agreed on a 2 week timeline"}}
  ],
  "participant_contributions": [
    {{"name": "John", "role": "Manager", "contributions": "Led the discussion and assigned tasks"}}
  ]
}}

Now extract the REAL action items, decisions, discussion points, and contributions from this transcript. Use actual names and details from the text. Return ONLY the JSON.

TRANSCRIPT:
{transcript}

JSON:"""


# ---- Read a transcript file ----
def read_transcript(filepath):
    """Read a .txt file and return its text."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ---- Send transcript to LLM and get structured output ----
def extract_insights(transcript_text):
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.1,
        num_predict=2048  # limit output length so small models don't loop forever
    )

    prompt = PromptTemplate(
        template=EXTRACTION_PROMPT,
        input_variables=["transcript"]
    )

    chain = prompt | llm

    print("Sending transcript to LLM for extraction...")
    raw_response = chain.invoke({"transcript": transcript_text})

    # try to parse the response as JSON
    parsed = parse_json_response(raw_response)
    return parsed


def parse_json_response(raw_text):
    """Try to pull valid JSON out of the LLM's response.
    LLMs sometimes produce slightly broken JSON, so we try
    multiple ways to salvage it."""
    text = raw_text.strip()

    # find the first { and last }
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end <= start:
        return {"error": "No JSON found in response", "raw_response": raw_text}

    json_str = text[start:end + 1]

    # try 1: parse directly
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # try 2: LLM sometimes puts broken text in the middle of a field.
    # remove lines that don't look like valid JSON (no quotes, no braces, no brackets)
    cleaned_lines = []
    for line in json_str.split("\n"):
        stripped = line.strip()
        if stripped == "":
            continue
        # keep lines that have JSON-like characters
        if any(c in stripped for c in ['"', '{', '}', '[', ']', ':']):
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # try 3: extract each section separately so we get partial data
    result = {}
    for key in ["action_items", "decisions", "key_discussion_points", "participant_contributions"]:
        try:
            # find the array for this key
            key_pos = json_str.find(f'"{key}"')
            if key_pos == -1:
                continue
            arr_start = json_str.find("[", key_pos)
            if arr_start == -1:
                continue
            # find matching closing bracket
            depth = 0
            for i in range(arr_start, len(json_str)):
                if json_str[i] == "[":
                    depth += 1
                elif json_str[i] == "]":
                    depth -= 1
                if depth == 0:
                    arr_str = json_str[arr_start:i + 1]
                    result[key] = json.loads(arr_str)
                    break
        except (json.JSONDecodeError, IndexError):
            continue

    if result:
        return result

    return {"error": "Could not parse JSON", "raw_response": raw_text}


# ---- Pretty print the results ----
def print_insights(insights):
    """Print the extracted insights in a readable format."""
    if "error" in insights:
        print(f"\nError: {insights['error']}")
        return

    for section in ["action_items", "decisions", "key_discussion_points", "participant_contributions"]:
        print(f"\n{'=' * 50}")
        print(section.upper().replace("_", " "))
        print("=" * 50)
        for item in insights.get(section, []):
            print(f"  {item}")


# ---- Full pipeline: file -> extract -> print ----

def extract_from_file(filepath):
    """Read a transcript file and extract structured insights."""
    print(f"Reading: {filepath}")
    text = read_transcript(filepath)
    print(f"  Transcript length: {len(text)} characters")

    insights = extract_insights(text)
    print_insights(insights)

    return insights


# ---- Run directly to test ----

if __name__ == "__main__":
    transcript_file = "./data/transcripts/sprint_planning.txt"
    result = extract_from_file(transcript_file)

    # also save the JSON to a file for reference
    output_path = "./data/insights_output.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nJSON saved to {output_path}")
