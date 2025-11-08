"""
Bitcoin Trading Analysis System using CrewAI
Analyzes recent Bitcoin articles and provides buy/sell recommendations
"""

import sys
import os
import json
import re
from datetime import date, datetime
from typing import Optional
from pathlib import Path

# Check Python version
MIN_PYTHON_VERSION = (3, 8)
if sys.version_info < MIN_PYTHON_VERSION:
    print(f"\n❌ Python version error!")
    print(f"   This script requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or higher.")
    print(f"   Your current version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"\n💡 To fix this:")
    print(f"   1. Install Python 3.8+ from https://www.python.org/downloads/")
    print(f"   2. Or use pyenv: pyenv install 3.11")
    print(f"   3. Or use conda: conda create -n bitcoin python=3.11")
    sys.exit(1)

# Try to import required packages with helpful error messages
try:
    from dotenv import load_dotenv
except ImportError:
    print("\n❌ Missing package: python-dotenv")
    print("   Install it with: pip install python-dotenv")
    sys.exit(1)

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool
except ImportError as e:
    print("\n❌ Missing package: crewai")
    print("   Install it with: pip install crewai")
    print(f"   Error details: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Create custom tools using the tool decorator
@tool("Search the web for recent articles")
def search_web_tool(query: str) -> str:
    """Search the web for recent articles using SerpAPI.
    
    Args:
        query: The search query string
        
    Returns:
        Search results as a string
    """
    try:
        import requests
        serpapi_key = os.getenv('SERPER_API_KEY')  # Using SERPER_API_KEY env var name for SerpAPI key
        if not serpapi_key or serpapi_key in ['your_serper_api_key_here', '']:
            return "Error: SERPER_API_KEY not set in .env file"
        
        # Use SerpAPI endpoint (serpapi.com)
        url = "https://serpapi.com/search.json"
        params = {
            'q': query,
            'api_key': serpapi_key,
            'num': 10,
            'engine': 'google'
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            results = []
            # SerpAPI uses 'organic_results' instead of 'organic'
            if 'organic_results' in data:
                for item in data['organic_results']:
                    title = item.get('title', 'N/A')
                    link = item.get('link', 'N/A')
                    snippet = item.get('snippet', item.get('about_this_result', {}).get('source', {}).get('description', 'N/A'))
                    results.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}\n")
            return "\n".join(results) if results else "No results found"
        elif response.status_code == 401 or response.status_code == 403:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get('error', 'Unauthorized')
            return f"Error: SerpAPI authentication failed ({response.status_code}). Please check your SERPER_API_KEY in .env file. The key may be invalid, expired, or your account may have exceeded its quota. Visit https://serpapi.com to verify your API key. Error: {error_msg}"
        elif response.status_code == 429:
            return f"Error: SerpAPI rate limit exceeded (429). Please wait a moment and try again, or check your quota at https://serpapi.com"
        else:
            error_detail = response.text[:200] if response.text else 'No details'
            return f"Error: SerpAPI returned status code {response.status_code}. Details: {error_detail}"
    except Exception as e:
        return f"Error searching: {str(e)}"

@tool("Read content from a website URL")
def read_website_tool(url: str) -> str:
    """Read and extract content from a website URL.
    
    Args:
        url: The website URL to read
        
    Returns:
        Website content as a string
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000] if len(text) > 5000 else text  # Limit to 5000 chars
    except Exception as e:
        return f"Error reading website: {str(e)}"

# Initialize tools
search_tool = search_web_tool
website_tool = read_website_tool

def _fallback_persona() -> dict:
    """Return the fixed AI investor profile (image stored locally in assets/)."""
    return {
        "name": "Avery S. Coinwright",
        "title": "Chief Investment Strategist",
        "bio": (
            "Avery S. Coinwright synthesizes crypto market structure, ETF flows, "
            "and macro currents to brief traders on intraday positioning."
        ),
        "image_src": "assets/ai-bitcoin-analyst.svg",
    }


def _ensure_reports_dir() -> Path:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def _collect_archive_items(current_date: str, max_items: int = 30) -> list:
    reports_dir = _ensure_reports_dir()
    items = []

    for html_path in reports_dir.glob("*.html"):
        try:
            report_date = datetime.strptime(html_path.stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        items.append(
            {
                "date": report_date,
                "href": f"reports/{html_path.name}",
            }
        )

    try:
        today_date = datetime.strptime(current_date, "%Y-%m-%d").date()
    except ValueError:
        today_date = date.today()

    if not any(entry["date"] == today_date for entry in items):
        items.append(
            {
                "date": today_date,
                "href": f"reports/{current_date}.html",
            }
        )

    items.sort(key=lambda entry: entry["date"], reverse=True)
    return items[:max_items]


def _render_archive_links(archive_items: list) -> tuple[str, str]:
    if not archive_items:
        return '<li class="empty-state">No past reports available yet.</li>', date.today().isoformat()

    list_items = []
    for item in archive_items:
        iso_date = item["date"].isoformat()
        list_items.append(f'<li><a href="{item["href"]}">{iso_date}</a></li>')

    min_date = archive_items[-1]["date"].isoformat()
    return "\n          ".join(list_items), min_date


def _save_daily_report_files(report_data: dict, html_output: str, current_date: str) -> None:
    reports_dir = _ensure_reports_dir()
    html_path = reports_dir / f"{current_date}.html"
    json_path = reports_dir / f"{current_date}.json"
    archive_html_output = html_output.replace('href="reports/', 'href="')

    html_path.write_text(archive_html_output, encoding="utf-8")
    json_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_recent_report_summaries(limit: int = 7, exclude_date: Optional[str] = None) -> list:
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return []

    entries = []
    for json_path in reports_dir.glob("*.json"):
        try:
            report_date = datetime.strptime(json_path.stem, "%Y-%m-%d").date()
        except ValueError:
            continue

        iso_date = report_date.isoformat()
        if exclude_date and iso_date == exclude_date:
            continue

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        entries.append({"date": report_date, "data": data})

    entries.sort(key=lambda entry: entry["date"], reverse=True)
    return entries[:limit]


def _build_history_context(recent_reports: list) -> str:
    if not recent_reports:
        return ""

    lines = []
    for entry in recent_reports:
        iso_date = entry["date"].isoformat()
        data = entry["data"]
        summary = data.get("summary", {})
        synthesis = summary.get("topline") or (data.get("market_synthesis") or [""])
        if isinstance(synthesis, list):
            synthesis_text = synthesis[0] if synthesis else ""
        else:
            synthesis_text = synthesis

        recommendation = summary.get("recommendation") or ""
        if not recommendation:
            paragraphs = (data.get("recommendation") or {}).get("paragraphs", [])
            recommendation = paragraphs[0] if paragraphs else ""

        if not synthesis_text and not recommendation:
            continue

        snippet_parts = []
        if synthesis_text:
            snippet_parts.append(synthesis_text.strip())
        if recommendation:
            snippet_parts.append(f"Recommendation: {recommendation.strip()}")

        if snippet_parts:
            lines.append(f"- {iso_date}: " + " | ".join(snippet_parts))

    return "\n".join(lines[: len(recent_reports)])


def generate_fake_investor() -> dict:
    """Return the fixed persona profile without calling external APIs."""
    persona = _fallback_persona()
    assets_dir = Path(persona["image_src"]).parent
    if str(assets_dir) not in ("", "."):
        assets_dir.mkdir(parents=True, exist_ok=True)
    return persona


class BitcoinAnalyzer:
    def __init__(self):
        self.setup_agents()
        self.setup_tasks()
        self.setup_crew()
    
    def setup_agents(self):
        """Initialize all CrewAI agents"""
        
        # Google Search Agent
        self.search_agent = Agent(
            role='Bitcoin News Researcher',
            goal='Find the most recent and relevant Bitcoin articles from the past 24 hours',
            backstory="""You are an expert researcher specializing in cryptocurrency news.
            You excel at finding the latest, most relevant articles about Bitcoin from
            reputable sources. You focus on recent news that could impact trading decisions.""",
            tools=[search_tool],
            verbose=True,
            allow_delegation=False
        )
        
        # Article Reader Agent
        self.reader_agent = Agent(
            role='Article Analyst',
            goal='Extract and summarize key information from Bitcoin articles',
            backstory="""You are a skilled financial journalist with deep understanding
            of cryptocurrency markets. You can quickly identify the most important
            information in articles, including price movements, market sentiment,
            technical analysis, and major news events.""",
            tools=[website_tool],
            verbose=True,
            allow_delegation=False
        )
        
        # Synthesis Agent
        self.synthesis_agent = Agent(
            role='Market Intelligence Synthesizer',
            goal='Combine multiple article summaries into coherent market insights',
            backstory="""You are a senior market analyst who excels at identifying patterns
            and trends across multiple sources. You can synthesize information from various
            articles to create a comprehensive view of the current Bitcoin market situation.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Analyst Agent
        self.analyst_agent = Agent(
            role='Trading Strategist',
            goal='Provide clear buy/sell/hold recommendations based on market analysis',
            backstory="""You are an experienced cryptocurrency trading strategist with
            a track record of successful market predictions. You analyze market data,
            sentiment, and trends to provide actionable trading recommendations.
            You are conservative and base recommendations on solid evidence.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Website Agent
        self.website_agent = Agent(
            role='Financial News Layout Editor',
            goal='Prepare clear sectioned HTML content for a professional financial article in the tone of major newspapers such as The New York Times.',
            backstory="""You are a meticulous financial news layout editor. You produce semantic, well-structured HTML
            containing only the essential sections of the Bitcoin report (Articles Found, Article Analysis & Summaries,
            Market Synthesis, Final Trading Recommendation). Avoid decorative styling, animations, or bright colors.
            Focus on clean markup (headings, paragraphs, lists) with informative text that can be restyled later.""",
            verbose=True,
            allow_delegation=False
        )
    
    def setup_tasks(self):
        """Define tasks for each agent"""
        
        default_topic = "Bitcoin market today trading analysis"
        self._search_task_template = """Search for the most recent articles about: {topic}
        Focus on articles from the past 24 hours. Retrieve up to 10 of the most relevant articles
        from reputable financial news sources, cryptocurrency news sites, and major news outlets.
        Include article titles, URLs, and brief descriptions."""

        self._synthesis_task_base = """Using the article summaries from the reader agent, combine all
        summaries into a comprehensive market analysis. Identify:
        1. Common themes and patterns across articles
        2. Overall market sentiment (bullish/bearish/neutral)
        3. Key price levels or trends mentioned
        4. Major catalysts or events
        5. Conflicting information or uncertainties
        6. Consensus views vs. outlier opinions

        Create a unified view of the current Bitcoin market situation."""

        self._analyst_task_base = """Based on the synthesized market analysis from the synthesis agent,
        provide a clear trading recommendation for TODAY:

        1. Recommendation: BUY, SELL, or HOLD
        2. Confidence level: High, Medium, or Low
        3. Key reasons supporting the recommendation
        4. Risk factors to consider
        5. Suggested entry/exit points (if applicable)
        6. Time horizon for the recommendation

        Be specific and actionable. Base your recommendation on the evidence
        from the articles analyzed."""

        # Task 1: Search for recent articles
        self.search_task = Task(
            description=self._search_task_template.format(topic=default_topic),
            agent=self.search_agent,
            expected_output="""A list of recent Bitcoin articles with:
            - Article titles
            - Source URLs
            - Brief descriptions
            - Publication dates (if available)"""
        )
        
        # Task 2: Read and summarize articles
        self.reader_task = Task(
            description="""Using the articles found by the search agent, read each article
            and extract and summarize:
            1. Main topic and key points
            2. Price movements mentioned
            3. Market sentiment (bullish/bearish/neutral)
            4. Technical indicators or analysis
            5. Major news events or catalysts
            6. Risk factors mentioned
            
            Create concise summaries (2-3 sentences per article) highlighting
            the most important trading-relevant information. Use the website tool
            to fetch the full content of each article URL.""",
            agent=self.reader_agent,
            context=[self.search_task],
            expected_output="""A structured summary for each article containing:
            - Key points
            - Market sentiment
            - Price implications
            - Risk factors"""
        )
        
        # Task 3: Synthesize information
        self.synthesis_task = Task(
            description=self._synthesis_task_base,
            agent=self.synthesis_agent,
            context=[self.reader_task],
            expected_output="""A comprehensive synthesis report with:
            - Overall market sentiment
            - Key themes and patterns
            - Price trends and levels
            - Major catalysts
            - Risk assessment"""
        )
        
        # Task 4: Provide trading recommendation
        self.analyst_task = Task(
            description=self._analyst_task_base,
            agent=self.analyst_agent,
            context=[self.synthesis_task],
            expected_output="""A clear trading recommendation with:
            - BUY/SELL/HOLD decision
            - Confidence level
            - Supporting reasons
            - Risk factors
            - Entry/exit guidance"""
        )
        
        # Task 5: Create structured HTML content for the report
        self.website_task = Task(
            description="""Produce clean, semantic HTML content (without inline styling) that contains the latest Bitcoin
            report in clearly marked sections with the following IDs:
            - #articles-found : includes an <h2> and an unordered list (<ul id="articles-list">) of up to 10 articles with anchors.
            - #article-analysis : includes an <h2> and a series of <article> elements summarizing each article.
            - #market-synthesis : includes an <h2> and several <p> elements summarising market synthesis points.
            - #final-recommendation : includes an <h2>, paragraphs, and bullet lists describing recommendation, confidence,
              reasons, risk factors, and suggested entry/exit points.
            Keep the tone professional and data-driven. Avoid decorative language and do not include CSS or JavaScript.
            The HTML will be restyled later, so focus on structure and clarity only.""",
            agent=self.website_agent,
            context=[self.search_task, self.reader_task, self.synthesis_task, self.analyst_task],
            expected_output="""Semantic HTML fragment with sections #articles-found, #article-analysis, #market-synthesis,
            and #final-recommendation, each containing descriptive headings, paragraphs, and lists with up-to-date analysis."""
        )
    
    def setup_crew(self):
        """Create the Crew with all agents and tasks"""
        self.crew = Crew(
            agents=[
                self.search_agent,
                self.reader_agent,
                self.synthesis_agent,
                self.analyst_agent,
                self.website_agent
            ],
            tasks=[
                self.search_task,
                self.reader_task,
                self.synthesis_task,
                self.analyst_task,
                self.website_task
            ],
            process=Process.sequential,
            verbose=True
        )
    
    def analyze(self, topic="Bitcoin market today"):
        """Run the analysis pipeline"""
        print(f"\n🔍 Starting analysis for: {topic}\n")
        print("=" * 60)
        
        today_str = date.today().isoformat()
        recent_reports = _load_recent_report_summaries(limit=7, exclude_date=today_str)
        history_context = _build_history_context(recent_reports)
        history_note = ""
        if history_context:
            history_note = "\n\nRecent Bitcoin market history from prior reports:\n" + history_context

        # Update search task with the topic
        self.search_task.description = self._search_task_template.format(topic=topic) + history_note
        self.synthesis_task.description = self._synthesis_task_base + history_note
        self.analyst_task.description = self._analyst_task_base + history_note
        
        # Execute the crew
        persona = generate_fake_investor()
        result = self.crew.kickoff()
        
        # Extract HTML from the result and save it
        self._save_html_output(result, persona, history_context)
        
        return result
    
    def _save_html_output(self, result, persona, history_context: str = ""):
        """Extract and save HTML output from the website agent"""
        try:
            # Get the output from the website task
            # CrewAI returns the last task's output as the main result
            website_output = str(result)
            
            # Try to extract HTML content if it's wrapped in markdown code blocks
            html_content = None
            
            # Check for HTML in markdown code blocks
            if "```html" in website_output:
                html_content = website_output.split("```html")[1].split("```")[0].strip()
            elif "```" in website_output:
                # Try to find HTML between code blocks
                parts = website_output.split("```")
                for part in parts:
                    if "<html" in part.lower() or "<!doctype" in part.lower() or "<body" in part.lower():
                        html_content = part.strip()
                        break
                if not html_content:
                    # Check if any part looks like HTML
                    for part in parts:
                        if part.strip().startswith("<") and len(part.strip()) > 100:
                            html_content = part.strip()
                            break
            
            # If no HTML found in code blocks, use the full output
            if not html_content:
                html_content = website_output.strip()
            
            # Clean up any markdown formatting
            html_content = html_content.replace("```html", "").replace("```", "").strip()
            
            # Ensure it's valid HTML (wrap if necessary)
            if not html_content.strip().startswith("<!DOCTYPE") and not html_content.strip().startswith("<html"):
                html_content = f"<!DOCTYPE html><html><body>{html_content}</body></html>"

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            current_date = date.today().isoformat()
            archive_items = _collect_archive_items(current_date)
            archive_list_html, archive_min_date = _render_archive_links(archive_items)

            def extract_articles() -> list:
                items = []
                article_list = soup.find(id="articles-list")
                if article_list:
                    for li in article_list.find_all("li"):
                        link = li.find("a")
                        title = link.get_text(strip=True) if link else li.get_text(strip=True)
                        href = link.get("href") if link else None
                        if title:
                            items.append({"title": title, "href": href})
                return items

            def extract_analysis() -> list:
                entries = []
                section = soup.find(id="article-analysis")
                if section:
                    for block in section.find_all(["article", "p"]):
                        text = block.get_text(" ", strip=True)
                        if text:
                            entries.append(text)
                return entries

            def extract_synthesis() -> list:
                paragraphs = []
                section = soup.find(id="market-synthesis")
                if section:
                    for p_tag in section.find_all("p"):
                        text = p_tag.get_text(" ", strip=True)
                        if text:
                            paragraphs.append(text)
                return paragraphs

            def extract_recommendation() -> dict:
                result_data = {"paragraphs": [], "lists": []}
                section = soup.find(id="final-recommendation")
                if not section:
                    return result_data
                for child in section.find_all(["p", "ul", "ol"], recursive=False):
                    if child.name == "p":
                        text = child.get_text(" ", strip=True)
                        if text:
                            result_data["paragraphs"].append(text)
                    elif child.name in ("ul", "ol"):
                        items = [li.get_text(" ", strip=True) for li in child.find_all("li")]
                        if items:
                            result_data["lists"].append(items)
                return result_data

            articles = extract_articles()
            analysis = extract_analysis()
            synthesis = extract_synthesis()
            recommendation = extract_recommendation()

            def shorten_text(text: str, max_chars: int = 160) -> str:
                cleaned = re.sub(r"\s+", " ", text or "").strip()
                if not cleaned:
                    return ""
                sentences = re.split(r"(?<=[.!?])\s+", cleaned)
                snippet = sentences[0] if sentences else cleaned
                if len(snippet) > max_chars:
                    snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "…"
                snippet = snippet.strip()
                if snippet and snippet[-1] not in ".!?…":
                    snippet += "."
                return snippet

            def unique_snippets(
                raw_items,
                max_items: int = 5,
                skip_keywords: Optional[set] = None,
                max_chars: int = 120,
            ) -> list:
                skip_keywords = skip_keywords or set()
                seen = set()
                snippets = []
                for raw in raw_items:
                    snippet = shorten_text(raw, max_chars=max_chars)
                    if not snippet:
                        continue
                    lowered = snippet.lower()
                    if any(keyword in lowered for keyword in skip_keywords):
                        continue
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    snippets.append(snippet)
                    if len(snippets) >= max_items:
                        break
                return snippets

            def format_labeled_sentence(label: str, snippet: str) -> str:
                cleaned = re.sub(r"[.…]+$", "", (snippet or "").strip()).strip()
                if not cleaned:
                    return ""
                if cleaned[-1] not in ".!?":
                    cleaned = f"{cleaned}."
                label = (label or "").strip()
                return f"{label} {cleaned}" if label else cleaned

            article_items_html = "\n".join(
                f'<li><a href="{item["href"]}" target="_blank" rel="noopener noreferrer">{item["title"]}</a></li>'
                if item["href"] else f'<li>{item["title"]}</li>'
                for item in articles
            ) or '<li>No recent articles were retrieved.</li>'

            unique_analysis_snippets = []
            seen_analysis = set()
            for text in analysis:
                snippet = shorten_text(text, max_chars=140)
                key = snippet.lower()
                if not snippet or key in seen_analysis:
                    continue
                seen_analysis.add(key)
                unique_analysis_snippets.append(snippet)

            analysis_entries = []
            max_articles = min(len(articles), len(unique_analysis_snippets), 6)
            transition_phrases = [
                "The report highlights that",
                "In addition, the analysis notes that",
                "Furthermore, coverage indicates that",
                "Another perspective explains that",
                "Market commentary confirms that",
                "Finally, observers point out that",
            ]
            for idx in range(max_articles):
                article = articles[idx]
                summary = unique_analysis_snippets[idx] if idx < len(unique_analysis_snippets) else ""
                if not summary:
                    continue
                title_html = (
                    f'<a href="{article["href"]}" target="_blank" rel="noopener noreferrer">{article["title"]}</a>'
                    if article["href"]
                    else article["title"]
                )
                lead_in = "According to"
                if idx < len(transition_phrases):
                    lead_in = transition_phrases[idx]
                paragraph = f"<p class=\"analysis-entry\"><span class=\"bullet-title\">{title_html}</span>: {lead_in} {summary}</p>"
                analysis_entries.append(paragraph)

            analysis_html = (
                "\n          ".join(analysis_entries)
                if analysis_entries
                else '<p class="empty-state">Analysis summaries are not available.</p>'
            )

            synthesis_sentences = []
            for paragraph in synthesis:
                synthesis_sentences.extend(re.split(r"(?<=[.!?])\s+", paragraph))
            synthesis_points = unique_snippets(synthesis_sentences, max_items=4)
            synthesis_points = [shorten_text(point, max_chars=220) for point in synthesis_points]
            if synthesis_points:
                synthesis_paragraphs = []
                connectors = [
                    "Overall,",
                    "In the near term,",
                    "From a structural standpoint,",
                    "Looking ahead,"
                ]
                for idx, point in enumerate(synthesis_points):
                    connector = connectors[idx] if idx < len(connectors) else "Additionally,"
                    synthesis_paragraphs.append(f"<p class=\"synthesis-entry\">{connector} {point}</p>")
                synthesis_html = "\n          ".join(synthesis_paragraphs)
            else:
                synthesis_html = '<p class="empty-state">Market synthesis is not available.</p>'

            recommendation_lines = recommendation.get("paragraphs", [])
            recommendation_summary = ""
            confidence_summary = ""
            for line in recommendation_lines:
                lowered = line.lower()
                if not recommendation_summary and "recommendation" in lowered:
                    recommendation_summary = shorten_text(line, max_chars=120)
                if not confidence_summary and "confidence" in lowered:
                    confidence_summary = shorten_text(line, max_chars=120)
            if not recommendation_summary:
                recommendation_summary = "Recommendation: Not provided"
            if not confidence_summary:
                confidence_summary = "Confidence Level: Not provided"

            recommendation_value = (
                recommendation_summary.split(":", 1)[1].strip() if ":" in recommendation_summary else recommendation_summary
            )
            confidence_value = (
                confidence_summary.split(":", 1)[1].strip() if ":" in confidence_summary else confidence_summary
            )
            recommendation_badge = f"Recommendation: {recommendation_value}"
            confidence_badge = f"Confidence Level: {confidence_value}"

            raw_rec_points = recommendation_lines[:]
            for lst in recommendation.get("lists", []):
                raw_rec_points.extend(lst)
            rec_points = unique_snippets(
                raw_rec_points,
                max_items=4,
                skip_keywords={
                    "recommendation:",
                    "confidence:",
                    "confidence level",
                    "key reasons",
                    "risk factors",
                    "suggested entry",
                    "suggested exit",
                    "time horizon",
                    "summary:",
                },
                max_chars=180,
            )
            focus_sentence = (
                format_labeled_sentence("Near-term focus:", rec_points[0])
                if rec_points
                else "Near-term focus: Monitor the $100K support and $106K resistance ranges closely."
            )
            follow_sentence = (
                format_labeled_sentence("Next steps:", rec_points[1])
                if len(rec_points) > 1
                else "Next steps: Stay nimble and update positioning once momentum confirms a clear break."
            )
            primary_sentence = (
                f"Today's call is to {recommendation_value.upper()} with {confidence_value.lower()} confidence."
            )
            recommendation_paragraph = " ".join(
                part.strip()
                for part in (primary_sentence, focus_sentence, follow_sentence)
                if part.strip()
            )
            recommendation_block = f'<p class="recommendation-summary">{recommendation_paragraph}</p>'

            synthesis_topline = synthesis_points[0] if synthesis_points else ""

            report_data = {
                "date": current_date,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "persona": persona,
                "articles": articles,
                "analysis_entries": analysis,
                "market_synthesis": synthesis,
                "recommendation": recommendation,
                "summary": {
                    "topline": synthesis_topline,
                    "recommendation": recommendation_paragraph,
                    "confidence": confidence_badge,
                    "badge_recommendation": recommendation_badge,
                    "badge_confidence": confidence_badge,
                    "recommendation_focus": focus_sentence,
                    "recommendation_next": follow_sentence,
                },
                "history_context": history_context,
            }

            persona_note = (
                f"Daily commentary by {persona['name']}, AI-generated fictional market analyst."
            )
            persona_image_src = persona.get("image_src") or _fallback_persona()["image_src"]

            html_output = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bitcoin Market Analysis Report</title>
  <style>
    :root {{
      color-scheme: light;
    }}
    body {{
      margin: 0;
      font-family: "Georgia", "Times New Roman", serif;
      background-color: #f8f7f5;
      color: #111111;
    }}
    a {{
      color: #0b63ce;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .page {{
      max-width: 760px;
      margin: 0 auto;
      padding: 3rem 1.5rem 4rem;
      background-color: #ffffff;
      min-height: 100vh;
    }}
    header.masthead {{
      border-bottom: 1px solid #e0e0e0;
      padding-bottom: 1.75rem;
      margin-bottom: 1.75rem;
    }}
    header.masthead h1 {{
      font-size: 2.4rem;
      line-height: 1.2;
      margin: 0 0 0.75rem 0;
      font-weight: 700;
      color: #111111;
    }}
    header.masthead p.subheading {{
      font-size: 1.1rem;
      line-height: 1.6;
      color: #4a4a4a;
      margin: 0;
    }}
    header.masthead p.report-date {{
      font-size: 1rem;
      line-height: 1.5;
      color: #333333;
      margin: 0.4rem 0 0;
    }}
    .headline-summary {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 1.75rem;
    }}
    .status-badge {{
      display: inline-flex;
      align-items: center;
      padding: 0.4rem 0.85rem;
      border-radius: 999px;
      font-size: 0.95rem;
      font-weight: 600;
      letter-spacing: 0.01em;
    }}
    .status-recommendation {{
      background-color: #1f5135;
      color: #f5fff8;
    }}
    .status-confidence {{
      background-color: #0c4e78;
      color: #edf7ff;
    }}
    .persona-block {{
      display: flex;
      gap: 1.5rem;
      border-bottom: 1px solid #e0e0e0;
      padding-bottom: 1.75rem;
      margin-bottom: 2rem;
      align-items: center;
    }}
    .persona-block img {{
      width: 108px;
      height: 108px;
      object-fit: cover;
      border-radius: 50%;
      background-color: #ddd;
      flex-shrink: 0;
    }}
    .persona-details {{
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }}
    .persona-name {{
      font-size: 1.25rem;
      font-weight: 700;
      color: #111111;
    }}
    .persona-title {{
      font-size: 1.05rem;
      color: #333333;
    }}
    .persona-bio {{
      font-size: 1rem;
      line-height: 1.6;
      color: #444444;
    }}
    .persona-note {{
      font-size: 0.95rem;
      color: #666666;
    }}
    main {{
      line-height: 1.65;
    }}
    section {{
      margin-bottom: 2.5rem;
    }}
    .archive-section {{
      border-top: 1px solid #e0e0e0;
      padding-top: 2rem;
    }}
    section h2 {{
      font-size: 1.6rem;
      font-weight: 700;
      color: #333333;
      border-bottom: 1px solid #e0e0e0;
      padding-bottom: 0.75rem;
      margin-bottom: 1.25rem;
    }}
    .article-list {{
      list-style: disc;
      padding-left: 1.4rem;
      margin: 0;
    }}
    .article-list li {{
      margin-bottom: 0.65rem;
      font-size: 1rem;
    }}
    .article-list li:hover {{
      background-color: #f3f3f3;
    }}
    .article-list li a {{
      display: inline-block;
      padding: 0.2rem 0;
    }}
    .archive-note {{
      font-size: 1rem;
      color: #4a4a4a;
      margin: 0 0 1rem 0;
    }}
    .archive-list {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    .archive-list li {{
      font-size: 1rem;
      margin-bottom: 0.4rem;
    }}
    .archive-list li a {{
      color: #0b63ce;
    }}
    .analysis-entry {{
      margin: 0 0 1.2rem 0;
      font-size: 1rem;
      line-height: 1.65;
      color: #2f3b48;
    }}
    .analysis-entry .bullet-title {{
      font-weight: 600;
      color: #102542;
    }}
    .analysis-entry a {{
      color: #0b63ce;
    }}
    .synthesis-entry {{
      margin: 0 0 1.1rem 0;
      font-size: 1rem;
      line-height: 1.65;
      color: #2f3b48;
    }}
    .recommendation-summary {{
      font-size: 1rem;
      line-height: 1.6;
      color: #2f3b48;
      margin: 0;
    }}
    .recommendation-box {{
      background-color: #fafafa;
      border: 1px solid #e0e0e0;
      padding: 1.5rem;
      font-size: 1rem;
    }}
    .email-signup {{
      border-top: 1px solid #e0e0e0;
      border-bottom: 1px solid #e0e0e0;
      padding: 1.75rem 0;
    }}
    .email-signup h3 {{
      font-size: 1.3rem;
      margin-top: 0;
      margin-bottom: 0.75rem;
      color: #333333;
    }}
    .email-form {{
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
    }}
    .email-form input[type="email"] {{
      flex: 1 1 280px;
      padding: 0.65rem 0.85rem;
      border: 1px solid #c8c8c8;
      border-radius: 4px;
      font-size: 1rem;
      font-family: "Georgia", "Times New Roman", serif;
      color: #111111;
      background-color: #ffffff;
    }}
    .email-form input[type="email"]:focus {{
      outline: 2px solid #0b63ce;
      outline-offset: 2px;
    }}
    .email-form button {{
      padding: 0.65rem 1.4rem;
      border: 1px solid #0b63ce;
      background-color: #0b63ce;
      color: #ffffff;
      font-size: 1rem;
      font-family: "Georgia", "Times New Roman", serif;
      border-radius: 4px;
      cursor: pointer;
    }}
    .email-form button:disabled {{
      background-color: #9fbce0;
      border-color: #9fbce0;
      cursor: not-allowed;
    }}
    #email-message {{
      margin-top: 0.75rem;
      font-size: 0.95rem;
      color: #333333;
    }}
    footer.disclaimer {{
      margin-top: 3rem;
      padding-top: 1.5rem;
      border-top: 1px solid #e0e0e0;
      font-size: 0.95rem;
      color: #555555;
      line-height: 1.6;
    }}
    @media (max-width: 640px) {{
      .persona-block {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .persona-block img {{
        width: 88px;
        height: 88px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="masthead">
      <h1>Bitcoin Market Analysis Report</h1>
      <p class="subheading">Daily analysis of bitcoin price action, market flows, and key risks.</p>
      <p class="report-date" id="report-date">Report Date: {current_date}</p>
    </header>

    <div class="headline-summary" aria-label="Today&#39;s trading stance">
      <span class="status-badge status-recommendation">{recommendation_badge}</span>
      <span class="status-badge status-confidence">{confidence_badge}</span>
    </div>

    <section class="persona-block" aria-label="AI Analyst Persona">
      <img src="{persona_image_src}" alt="{persona['name']} headshot portrait" />
      <div class="persona-details">
        <p class="persona-name">{persona['name']}</p>
        <p class="persona-title">{persona['title']}</p>
        <p class="persona-bio">{persona['bio']}</p>
        <p class="persona-note">{persona_note}</p>
      </div>
    </section>

    <section class="archive-section" aria-label="Past Bitcoin Reports" tabindex="0">
      <h2>Recent Reports</h2>
      <p class="archive-note">Browse prior daily briefings for continuity in market context.</p>
      <ul class="archive-list">
          {archive_list_html}
      </ul>
    </section>

    <main>
      <section id="article-analysis" aria-label="Article Analysis and Summaries" tabindex="0">
        <h2>Article Analysis &amp; Summaries</h2>
        {analysis_html}
      </section>

      <section id="market-synthesis" aria-label="Complete Market Synthesis Report" tabindex="0">
        <h2>Market Synthesis</h2>
        {synthesis_html}
      </section>

      <section id="final-recommendation" aria-label="Final Trading Recommendation" tabindex="0">
        <h2>Final Trading Recommendation</h2>
        <div class="recommendation-box">
          {recommendation_block}
        </div>
      </section>

      <section class="email-signup" id="email-section" aria-label="Email Report Section">
        <h3>Receive the Report</h3>
        <p>Enter your email address to have the daily summary delivered to your inbox.</p>
        <div class="email-form">
          <input aria-describedby="email-message" aria-required="true" autocomplete="email" id="user-email" placeholder="Enter your email address" required type="email" />
          <button aria-busy="false" aria-live="polite" id="send-report-btn" onclick="sendReport()" disabled>Send Report</button>
        </div>
        <div aria-live="assertive" id="email-message" role="alert"></div>
      </section>

      <section id="articles-found" aria-label="Articles Found" tabindex="0">
        <h2>Articles Found</h2>
        <ul class="article-list">
          {article_items_html}
        </ul>
      </section>
    </main>

    <footer class="disclaimer">
      <p>This report and analyst persona are AI generated for informational and educational purposes only and do not constitute financial advice. Always perform independent research or consult a licensed financial professional before making investment decisions.</p>
    </footer>
  </div>

  <script>
    function updateReportDate() {{
      const dateElem = document.getElementById('report-date');
      if (!dateElem) return;
      const now = new Date();
      const options = {{ year: 'numeric', month: 'long', day: 'numeric' }};
      const formattedDate = now.toLocaleDateString('en-US', options);
      dateElem.textContent = 'Report Date: ' + formattedDate;
    }}
    updateReportDate();

    const emailInput = document.getElementById('user-email');
    const sendBtn = document.getElementById('send-report-btn');
    const emailMsg = document.getElementById('email-message');

    function validateEmail(email) {{
      const re = /^[a-zA-Z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{{2,}}$/i;
      return re.test(String(email).toLowerCase());
    }}

    window.sendReport = async function() {{
      const email = emailInput.value.trim();
      if (!validateEmail(email)) {{
        emailMsg.textContent = 'Please enter a valid email address!';
        emailMsg.style.color = '#b70000';
        return;
      }}

      sendBtn.disabled = true;
      sendBtn.textContent = 'Sending...';
      emailMsg.textContent = 'Sending report...';
      emailMsg.style.color = '#0f3c73';

      try {{
        const response = await fetch('http://localhost:5050/send-report', {{
          method: 'POST',
          headers: {{
            'Content-Type': 'application/json',
          }},
          body: JSON.stringify({{ email }})
        }});
        const data = await response.json();
        if (data.success) {{
          emailMsg.textContent = 'Report sent successfully!';
          emailMsg.style.color = '#146414';
          emailInput.value = '';
        }} else {{
          emailMsg.textContent = 'Error: ' + (data.error || 'Failed to send report');
          emailMsg.style.color = '#b70000';
        }}
      }} catch (error) {{
        emailMsg.textContent = 'Error: Could not connect to server. Make sure the email API is running.';
        emailMsg.style.color = '#b70000';
      }} finally {{
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send Report';
      }}
    }};

    emailInput.addEventListener('input', () => {{
      const isValid = validateEmail(emailInput.value.trim());
      sendBtn.disabled = !isValid;
      if (!isValid && emailInput.value.trim().length > 0) {{
        emailMsg.textContent = 'Please enter a valid email address!';
        emailMsg.style.color = '#b70000';
      }} else {{
        emailMsg.textContent = '';
      }}
    }});

    window.addEventListener('load', () => {{
      emailInput.focus();
    }});
  </script>
</body>
</html>
"""

            output_path = os.path.join(os.getcwd(), "index.html")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_output)

            _save_daily_report_files(report_data, html_output, current_date)
            
            print(f"\n✨ HTML report saved to: {output_path}")
            print("   Email API reminder: EMAIL_API_PORT=5050 python3.11 email_api.py")
            
        except Exception as e:
            print(f"\n⚠️  Warning: Could not save HTML output: {e}")
            print("The analysis completed successfully, but HTML generation had an issue.")
            import traceback
            traceback.print_exc()


def check_environment():
    """Check if the environment is properly set up"""
    issues = []
    
    # Check Python version (already checked at import, but double-check)
    if sys.version_info < MIN_PYTHON_VERSION:
        issues.append(f"Python version {sys.version_info.major}.{sys.version_info.minor} is too old (need {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+)")
    
    # Check for required API keys
    required_keys = ['OPENAI_API_KEY', 'SERPER_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key) or os.getenv(key).strip() in ['', 'your_openai_api_key_here', 'your_serper_api_key_here']]
    
    if missing_keys:
        issues.append(f"Missing API keys: {', '.join(missing_keys)}")
    
    return len(issues) == 0, issues


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("🚀 Bitcoin Trading Analysis System")
    print("=" * 60)
    
    # Check environment
    env_ok, issues = check_environment()
    
    if not env_ok:
        print("\n❌ Environment check failed!")
        for issue in issues:
            print(f"   - {issue}")
        
        # Provide helpful guidance
        if any("API keys" in issue for issue in issues):
            print("\n💡 To fix API keys:")
            print("   1. Edit your .env file")
            print("   2. Get OpenAI API key: https://platform.openai.com/api-keys")
            print("   3. Get Serper API key: https://serper.dev (free tier available)")
        
        if any("Python version" in issue for issue in issues):
            print("\n💡 To fix Python version:")
            print("   1. Install Python 3.8+ from https://www.python.org/downloads/")
            print("   2. Or use pyenv: pyenv install 3.11")
            print("   3. Or use conda: conda create -n bitcoin python=3.11")
        
        print("\n")
        return
    
    # Initialize analyzer
    analyzer = BitcoinAnalyzer()
    
    # Run analysis
    topic = "Bitcoin market today trading analysis"
    result = analyzer.analyze(topic)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 60)
    print("\n" + str(result))
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

