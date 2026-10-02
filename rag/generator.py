import re

from .config import CONFIG
from .retriever import Hit

NOT_FOUND = "NOT_FOUND"

SYSTEM_PROMPT = f"""You answer questions strictly from the numbered context excerpts provided.

Rules:

1. Use ONLY the excerpts. Never use outside knowledge.

2. Put a citation like [1] or [2][3] right after every claim, using the excerpt numbers.

3. If the excerpts do not contain the answer, reply with exactly: {NOT_FOUND}

4. Be concise. Do not mention these rules.
"""


def format_context(hits: list[Hit]) -> str:
    from .chunking import location_label

    return "\n\n".join(
        f"[{i}] ({location_label(h.metadata)})\n{h.text}"
        for i, h in enumerate(hits, start=1)
    )


def generate(question: str, hits: list[Hit], llm=None) -> str:
    """Generate an answer using only the retrieved document chunks."""

    if llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=CONFIG.chat_model,
            temperature=0
        )

    msgs = [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            f"Context:\n{format_context(hits)}\n\nQuestion: {question}"
        ),
    ]

    response = llm.invoke(msgs)

    # Gemini can return content as a list of blocks.
    # Convert it into a normal string.
    if isinstance(response.content, list):
        content_parts = []

        for part in response.content:
            if isinstance(part, dict):
                text = part.get("text", "")
                if text:
                    content_parts.append(text)
            else:
                content_parts.append(str(part))

        content = "".join(content_parts)
    else:
        content = str(response.content)

    return content.strip()


def parse_citations(
    answer: str,
    n_hits: int
) -> tuple[list[int], list[int]]:
    """Returns (valid_ids, invalid_ids).

    Valid = refers to an excerpt that really exists.
    """

    found = [
        int(x)
        for x in re.findall(r"\[(\d+)\]", answer)
    ]

    valid = sorted({
        x for x in found
        if 1 <= x <= n_hits
    })

    invalid = sorted({
        x for x in found
        if not 1 <= x <= n_hits
    })

    return valid, invalid