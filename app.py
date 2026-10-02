def web_search(query, max_results=5):
    try:
        print("WEB SEARCH:", query, flush=True)

        with DDGS(timeout=20) as ddgs:
            results = list(
                ddgs.text(
                    query,
                    region="us-en",
                    safesearch="moderate",
                    max_results=max_results
                )
            )

        if not results:
            print("WEB SEARCH: No results found", flush=True)
            return ""

        formatted = []

        for result in results:
            title = result.get("title", "")
            body = result.get("body", "")
            url = result.get("href", "")

            formatted.append(
                f"Title: {title}\n"
                f"Summary: {body}\n"
                f"Source: {url}"
            )

        print(
            "WEB SEARCH SUCCESS:",
            len(formatted),
            "results",
            flush=True
        )

        return "\n\n".join(formatted)

    except Exception as e:
        print(
            "WEB SEARCH ERROR:",
            repr(e),
            flush=True
        )
        return ""
