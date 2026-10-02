def web_search(query, max_results=5):
    try:
        print("WEB SEARCH:", query, flush=True)

        import requests

        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": os.environ["TAVILY_API_KEY"],
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": True
            },
            timeout=30
        )

        response.raise_for_status()
        data = response.json()

        formatted = []

        answer = data.get("answer", "")
        if answer:
            formatted.append(
                f"Answer: {answer}"
            )

        for result in data.get("results", []):
            title = result.get("title", "")
            body = result.get("content", "")
            url = result.get("url", "")

            formatted.append(
                f"Title: {title}\n"
                f"Summary: {body}\n"
                f"Source: {url}"
            )

        if not formatted:
            print("WEB SEARCH: No results found", flush=True)
            return ""

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
