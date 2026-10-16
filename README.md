# DISGEX

This repository contains the core code for **Schema Discovery for Clinical Information Extraction**.

**DISGEX** stands for *DIscovered-Schema-Guided EXtraction*. It discovers how clinical records are organized and uses that structure to guide information extraction.

## Method

DISGEX represents a record at three levels: the document, its context instances, and the function instances within each context. Context roles describe types of discussion, such as diagnostic evaluation. Function roles describe what a passage contributes, such as requesting an examination. Repeated discussions receive separate instances, each grounded in its own source spans. The code calls context roles *containers*.

1. **Discover the schema.** Learn shared context and function roles from unlabeled training records, with definitions and rules for when discussions begin, continue, or restart. The schema is fixed before evaluation records are parsed.
2. **Parse each record.** Locate role instances in the text and attach each function instance to its enclosing context. For nursing dictations, also recover source-grounded attributes describing attribution, time, and frequency.
3. **Guide extraction.** Provide the complete record and its structure through a task interface. The hierarchy helps the reader interpret evidence while the original wording remains available.

## Tasks

| Task | Task inputs | How DISGEX uses source structure |
| --- | --- | --- |
| CRF | Clinical notes and case report form definitions. | Map source roles to form fields and use these hints to fill case report forms. |
| SIMORD | Dialogues with ordered turns and source spans, plus example orders. | Select exact evidence passages and relevant context for medical orders, reasons, and supporting turn references. |
| SYNUR | Nursing dictations, an observation ontology, and training examples. | Review observation candidates using the hierarchy and its attributes, correcting values, removing unsupported outputs, and adding supported observations. |
| Stupp | Assessment-and-plan notes. | Discover and review schemas, then recover the hierarchical structure of the notes. |

## Code organization

| File | Purpose |
| --- | --- |
| [discovery.py](discovery.py) | Schema discovery, context boundary learning, and Stupp schema review and parsing. |
| [parsing.py](parsing.py) | Source segmentation, hierarchical parsing, span validation, and SYNUR schema and qualifier discovery. |
| [extraction.py](extraction.py) | Task readers, role-to-field mapping, SIMORD discovery and evidence selection, and SYNUR retrieval and verification. |
| [prompts.py](prompts.py) | Task prompts, learned schemas, qualifier definitions, and request settings. |
| [llm.py](llm.py) | Model request adapters and response decoding. |

## Run the CRF source parser

Use Python 3.10 or later and run the commands from this repository directory. Save a clinical note as UTF-8 text in `note.txt`. Set the chat-completions endpoint and API key for your inference service. The service must support the bundled request settings, including structured JSON responses.

This example applies the included CRF schema and prints the note with its parsed role hierarchy. It needs only Python's standard library and the repository modules. Schema discovery and final form extraction are separate steps.

```bash
export CHAT_COMPLETIONS_URL="https://your-api-server/v1/chat/completions"
export API_KEY="your-api-key"

python - <<'PY'
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import llm
import parsing

def call(request, purpose):
    req = Request(
        os.environ["CHAT_COMPLETIONS_URL"],
        data=json.dumps(request).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + os.environ["API_KEY"],
            "Content-Type": "application/json",
        },
    )
    with urlopen(req, timeout=120) as response:
        return json.load(response)

branch = parsing.crf()
api = llm.crf_api(SimpleNamespace(call=call), branch)
doc = {"document_id": "example", "clinical_note": Path("note.txt").read_text(encoding="utf-8")}
source = branch.ours.prepare_source(doc, api)
if not source["valid"]:
    raise RuntimeError("Source parsing failed validation.")
print(branch.raw.render(doc, source))
PY
```

The output retains the original note and shows the context and function instances with their source ranges. To parse another CRF note, replace `note.txt`.

For another task, the HTTP chat sender can be reused, but the CRF setup, input preparation, and parsing and output calls must be replaced with that task's code. Use the task inputs listed above and the corresponding request handling in [llm.py](llm.py). For SIMORD and SYNUR, use [parsing.py](parsing.py) and [extraction.py](extraction.py). For Stupp, use [discovery.py](discovery.py). SIMORD and Stupp also require a task schema. SYNUR retrieval also uses embeddings. The HTTP sender above handles chat requests only, so embedding requests need their own endpoint.

Final extraction uses the task interfaces in [extraction.py](extraction.py), with task data and reader components supplied by the application. Its dependencies and setup are separate from this standard-library parsing example.
