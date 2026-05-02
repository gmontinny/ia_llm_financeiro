from deep_translator import GoogleTranslator


def format_response(res: str, return_thinking: bool = False) -> str:
    res = res.strip()
    if return_thinking:
        res = res.replace("<think>", "[pensando...] ")
        res = res.replace("</think>", "\n---\n")
    else:
        if "</think>" in res:
            res = res.split("</think>")[-1].strip()
    return res


def translate(text: str, source_lang: str = "pt", target_lang: str = "en") -> str:
    return GoogleTranslator(source=source_lang, target=target_lang).translate(text)


def summarize_doc(llm, content: str) -> str:
    from src.config import SUMMARY_TEMPLATE

    prompt = SUMMARY_TEMPLATE.format(content)
    output = llm.complete(prompt)
    return format_response(output.text)
