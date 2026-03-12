#!/usr/bin/env python3
"""
Wikipedia search script using the MediaWiki API.
Outputs JSON results to stdout, errors to stderr.
"""

import argparse
import json
import subprocess
import sys

# Try to import wikipediaapi, auto-install if missing
try:
    import wikipediaapi
except ImportError:
    print("Installing Wikipedia-API library...", file=sys.stderr)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "Wikipedia-API",
             "--break-system-packages", "--quiet"],
            stderr=subprocess.DEVNULL
        )
        import wikipediaapi
        print("Installation complete.", file=sys.stderr)
    except Exception as e:
        print(f"Error: Failed to install Wikipedia-API: {e}", file=sys.stderr)
        print("Please install manually: pip install Wikipedia-API", file=sys.stderr)
        sys.exit(2)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Search and fetch content from Wikipedia"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search term or page title"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["search", "summary", "full"],
        default="summary",
        help="Operation mode: search for titles, get summary, or get full content (default: summary)"
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=5,
        help="Number of sentences for summary mode (default: 5)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code (default: en)"
    )

    return parser.parse_args()


def search_wikipedia(query, lang="en", limit=10):
    """
    Search for Wikipedia page titles matching the query.

    Args:
        query: Search term
        lang: Language code
        limit: Maximum number of results

    Returns:
        Dictionary with search results
    """
    try:
        wiki = wikipediaapi.Wikipedia(
            user_agent='OpenClaw-WikipediaSearch/1.0',
            language=lang
        )

        # Wikipedia-API doesn't have a direct search method
        # We'll try to get the page and use its links as suggestions
        # For a proper search, we need to use the MediaWiki search API
        import urllib.request
        import urllib.parse

        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            'action': 'opensearch',
            'search': query,
            'limit': limit,
            'format': 'json'
        }

        url = f"{search_url}?{urllib.parse.urlencode(params)}"

        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())

        # OpenSearch returns: [query, [titles], [descriptions], [urls]]
        if len(data) >= 4:
            titles = data[1]
            descriptions = data[2]
            urls = data[3]

            results = []
            for i in range(len(titles)):
                results.append({
                    "title": titles[i],
                    "description": descriptions[i] if i < len(descriptions) else "",
                    "url": urls[i] if i < len(urls) else ""
                })

            return {
                "mode": "search",
                "query": query,
                "results": results,
                "count": len(results)
            }
        else:
            return {
                "mode": "search",
                "query": query,
                "results": [],
                "count": 0
            }

    except Exception as e:
        print(f"Error during Wikipedia search: {e}", file=sys.stderr)
        return {
            "mode": "search",
            "query": query,
            "results": [],
            "count": 0,
            "error": str(e)
        }


def get_summary(title, lang="en", sentences=5):
    """
    Get a summary of a Wikipedia page.

    Args:
        title: Page title
        lang: Language code
        sentences: Number of sentences to return

    Returns:
        Dictionary with page summary
    """
    try:
        wiki = wikipediaapi.Wikipedia(
            user_agent='OpenClaw-WikipediaSearch/1.0',
            language=lang
        )

        page = wiki.page(title)

        if not page.exists():
            return {
                "mode": "summary",
                "title": title,
                "exists": False,
                "error": "Page not found"
            }

        # Get summary text
        summary_text = page.summary[0:sentences * 200] if page.summary else ""

        # Try to limit to requested number of sentences
        if summary_text:
            sentences_list = summary_text.replace('!', '.').replace('?', '.').split('.')
            sentences_list = [s.strip() for s in sentences_list if s.strip()]
            summary_text = '. '.join(sentences_list[:sentences]) + '.'

        return {
            "mode": "summary",
            "title": page.title,
            "exists": True,
            "url": page.fullurl,
            "summary": summary_text,
            "categories": [cat for cat in page.categories.keys()][:10]  # Limit categories
        }

    except Exception as e:
        print(f"Error fetching Wikipedia summary: {e}", file=sys.stderr)
        return {
            "mode": "summary",
            "title": title,
            "exists": False,
            "error": str(e)
        }


def get_full_content(title, lang="en"):
    """
    Get the full content of a Wikipedia page.

    Args:
        title: Page title
        lang: Language code

    Returns:
        Dictionary with full page content
    """
    try:
        wiki = wikipediaapi.Wikipedia(
            user_agent='OpenClaw-WikipediaSearch/1.0',
            language=lang
        )

        page = wiki.page(title)

        if not page.exists():
            return {
                "mode": "full",
                "title": title,
                "exists": False,
                "error": "Page not found"
            }

        # Get sections recursively
        def get_sections(section, level=0):
            sections_data = []
            for s in section.sections:
                section_info = {
                    "title": s.title,
                    "level": level,
                    "text": s.text[:5000] if s.text else ""  # Limit section text
                }
                sections_data.append(section_info)
                # Recursively get subsections
                sections_data.extend(get_sections(s, level + 1))
            return sections_data

        return {
            "mode": "full",
            "title": page.title,
            "exists": True,
            "url": page.fullurl,
            "summary": page.summary,
            "sections": get_sections(page),
            "categories": [cat for cat in page.categories.keys()][:10],
            "links": list(page.links.keys())[:50]  # Limit to first 50 links
        }

    except Exception as e:
        print(f"Error fetching Wikipedia content: {e}", file=sys.stderr)
        return {
            "mode": "full",
            "title": title,
            "exists": False,
            "error": str(e)
        }


def main():
    """Main entry point."""
    args = parse_args()

    # Validate query
    if not args.query or not args.query.strip():
        print("Error: Query cannot be empty", file=sys.stderr)
        sys.exit(1)

    # Execute based on mode
    if args.mode == "search":
        result = search_wikipedia(args.query, lang=args.lang)
    elif args.mode == "summary":
        result = get_summary(args.query, lang=args.lang, sentences=args.sentences)
    elif args.mode == "full":
        result = get_full_content(args.query, lang=args.lang)
    else:
        print(f"Error: Invalid mode '{args.mode}'", file=sys.stderr)
        sys.exit(1)

    # Output JSON
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Exit with appropriate code
    if "error" in result:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
