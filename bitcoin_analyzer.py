"""
Bitcoin Trading Analysis System using CrewAI
Analyzes recent Bitcoin articles and provides buy/sell recommendations
"""

import sys
import os

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
            role='SEVENTEEN-Inspired Web Designer',
            goal='Create a stunning SEVENTEEN-themed HTML report that displays all analysis results with K-pop boy group SEVENTEEN\'s energetic, synchronized, and colorful aesthetic',
            backstory="""You are a web designer inspired by SEVENTEEN, the 13-member K-pop boy group known for their
            synchronized performances, bright colorful aesthetics, and energetic vibe. You create websites that capture
            SEVENTEEN's signature style: vibrant colors (cyan, pink, purple, orange), clean modern design, smooth
            synchronized animations, and an upbeat, positive energy. You use SEVENTEEN's catchphrases like 'Fighting!',
            'Let's go!', and their positive, encouraging tone. You make everything look fresh, dynamic, and perfectly
            coordinated - just like SEVENTEEN's performances. The design should feel energetic, youthful, and full of
            positive vibes while maintaining professionalism.""",
            verbose=True,
            allow_delegation=False
        )
    
    def setup_tasks(self):
        """Define tasks for each agent"""
        
        # Task 1: Search for recent articles
        self.search_task = Task(
            description="""Search for the most recent Bitcoin articles from the past 24 hours.
            Focus on articles from reputable financial news sources, cryptocurrency news sites,
            and major news outlets. Retrieve up to 10 of the most relevant articles.
            Include article titles, URLs, and brief descriptions.""",
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
            description="""Using the article summaries from the reader agent, combine all
            summaries into a comprehensive market analysis. Identify:
            1. Common themes and patterns across articles
            2. Overall market sentiment (bullish/bearish/neutral)
            3. Key price levels or trends mentioned
            4. Major catalysts or events
            5. Conflicting information or uncertainties
            6. Consensus views vs. outlier opinions
            
            Create a unified view of the current Bitcoin market situation.""",
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
            description="""Based on the synthesized market analysis from the synthesis agent,
            provide a clear trading recommendation for TODAY:
            
            1. Recommendation: BUY, SELL, or HOLD
            2. Confidence level: High, Medium, or Low
            3. Key reasons supporting the recommendation
            4. Risk factors to consider
            5. Suggested entry/exit points (if applicable)
            6. Time horizon for the recommendation
            
            Be specific and actionable. Base your recommendation on the evidence
            from the articles analyzed.""",
            agent=self.analyst_agent,
            context=[self.synthesis_task],
            expected_output="""A clear trading recommendation with:
            - BUY/SELL/HOLD decision
            - Confidence level
            - Supporting reasons
            - Risk factors
            - Entry/exit guidance"""
        )
        
        # Task 5: Create HTML report
        self.website_task = Task(
            description="""Create a stunning, SEVENTEEN-inspired HTML report (index.html) that includes:
            
            1. All article names/titles from the search results
            2. All article analyses and summaries from the reader agent
            3. The complete synthesis report from the synthesis agent
            4. The final trading recommendation from the analyst agent
            
            Design requirements (SEVENTEEN aesthetic):
            - Vibrant, bright color scheme inspired by SEVENTEEN: cyan (#00D9FF), pink (#FF69B4), purple (#9B59B6), orange (#FF8C00)
            - Clean, modern design with synchronized animations (like SEVENTEEN's choreography)
            - Energetic, positive tone throughout - use SEVENTEEN's catchphrases like 'Fighting!', 'Let's go!', 'We're going to make it!'
            - Make it visually stunning with colorful gradients, smooth synchronized animations, and modern CSS
            - Include sections for: Articles Found, Article Analysis, Market Synthesis, Final Recommendation
            - Make the recommendation section stand out with a big, bold, energetic display
            - Add synchronized animations and hover effects (like SEVENTEEN's synchronized dance moves)
            - Use modern, clean fonts (like SEVENTEEN's clean aesthetic)
            - Bright, energetic color palette with perfect coordination
            - Make it mobile-responsive
            - Positive, encouraging tone throughout - make it feel like SEVENTEEN's positive energy
            
            CRITICAL REQUIREMENTS:
            - Display the current date prominently at the top of the page (update daily automatically using JavaScript)
            - Add an email section with:
              * A text input box for the user's email address
              * A "Send Report" button that sends the report via email
              * JavaScript to handle the email sending (connect to backend/API endpoint)
              * Validation for email format
              * Success/error messages after sending
            - The date should be displayed in a clear, visible format (e.g., "Report Date: November 6, 2024")
            - The email functionality should be integrated seamlessly into the SEVENTEEN design
            
            Write the complete HTML file with embedded CSS and JavaScript.
            The HTML should be a complete, standalone file that captures SEVENTEEN's vibrant, energetic, synchronized aesthetic.
            Include JavaScript to display the current date and handle email sending functionality.""",
            agent=self.website_agent,
            context=[self.search_task, self.reader_task, self.synthesis_task, self.analyst_task],
            expected_output="""A complete HTML file (index.html) with:
            - SEVENTEEN-inspired vibrant color design (cyan, pink, purple, orange)
            - All article names displayed
            - All analyses and summaries
            - Complete synthesis report
            - Final recommendation prominently displayed
            - Current date displayed prominently (updates daily)
            - Email input box and "Send Report" button
            - Email sending functionality with JavaScript
            - SEVENTEEN's positive, energetic tone throughout (Fighting!, Let's go!)
            - Synchronized animations and modern, clean styling"""
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
        
        # Update search task with the topic
        self.search_task.description = f"""Search for the most recent articles about: {topic}
        Focus on articles from the past 24 hours. Retrieve up to 10 of the most relevant articles
        from reputable financial news sources, cryptocurrency news sites, and major news outlets.
        Include article titles, URLs, and brief descriptions."""
        
        # Execute the crew
        result = self.crew.kickoff()
        
        # Extract HTML from the result and save it
        self._save_html_output(result)
        
        return result
    
    def _add_email_form_and_date(self, html_content):
        """Add email form and date display to HTML content"""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Add date display script
            date_script = soup.new_tag('script')
            date_script.string = """
            // Display current date
            function updateDate() {
                const now = new Date();
                const options = { year: 'numeric', month: 'long', day: 'numeric' };
                const dateString = now.toLocaleDateString('en-US', options);
                const dateElement = document.getElementById('report-date');
                if (dateElement) {
                    dateElement.textContent = 'Report Date: ' + dateString;
                }
            }
            updateDate();
            """
            soup.head.append(date_script)
            
            # Add email form script
            email_script = soup.new_tag('script')
            email_script.string = """
            async function sendReport() {
                const emailInput = document.getElementById('user-email');
                const email = emailInput.value.trim();
                const statusDiv = document.getElementById('email-status');
                
                // Validate email
                const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/;
                if (!email || !emailPattern.test(email)) {
                    statusDiv.textContent = 'Please enter a valid email address';
                    statusDiv.style.color = '#FF8FC7'; // WCAG AA compliant lighter pink
                    return;
                }
                
                // Disable button and show loading
                const sendBtn = document.getElementById('send-btn');
                sendBtn.disabled = true;
                sendBtn.textContent = 'Sending...';
                statusDiv.textContent = 'Sending report...';
                statusDiv.style.color = '#4DD0E1'; // WCAG AA compliant lighter cyan
                
                try {
                    const response = await fetch('http://localhost:5050/send-report', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ email: email })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        statusDiv.textContent = '✅ Report sent successfully!';
                        statusDiv.style.color = '#51cf66'; // High contrast green
                        emailInput.value = '';
                    } else {
                        statusDiv.textContent = '❌ Error: ' + (data.error || 'Failed to send report');
                        statusDiv.style.color = '#FF8FC7'; // WCAG AA compliant lighter pink
                    }
                } catch (error) {
                    statusDiv.textContent = '❌ Error: Could not connect to server. Make sure the email API is running.';
                    statusDiv.style.color = '#FF8FC7'; // WCAG AA compliant lighter pink
                } finally {
                    sendBtn.disabled = false;
                    sendBtn.textContent = 'Send Report';
                }
            }
            """
            soup.head.append(email_script)
            
            # Find body and add date display and email form at the top
            body = soup.body
            if body:
                # Create date display with WCAG AA compliant colors
                date_div = soup.new_tag('div', id='report-date')
                date_div['style'] = 'text-align: center; font-size: 1.2rem; font-weight: bold; margin: 20px 0; padding: 15px; background: rgba(30, 30, 35, 0.8); border-radius: 10px; color: #FFB74D;'  # WCAG AA compliant
                date_div.string = 'Report Date: Loading...'
                
                # Create email form section with WCAG AA compliant colors
                email_section = soup.new_tag('section', id='email-section')
                email_section['style'] = 'max-width: 600px; margin: 30px auto; padding: 25px; background: rgba(30, 30, 35, 0.8); border-radius: 15px; box-shadow: 0 0 20px rgba(255, 143, 199, 0.3);'
                
                email_title = soup.new_tag('h2')
                email_title.string = '📧 Send Report via Email'
                email_title['style'] = 'color: #FF8FC7; text-align: center; margin-bottom: 20px;'  # WCAG AA compliant lighter pink
                email_section.append(email_title)
                
                email_form = soup.new_tag('div')
                email_form['style'] = 'display: flex; flex-direction: column; gap: 15px;'
                
                email_input = soup.new_tag('input')
                email_input['type'] = 'email'
                email_input['id'] = 'user-email'
                email_input['placeholder'] = 'Enter your email address'
                email_input['style'] = 'padding: 12px; font-size: 1rem; border: 2px solid #4DD0E1; border-radius: 8px; background: rgba(255, 255, 255, 0.95); color: #1A1A1A; outline: none;'  # Light bg, dark text for WCAG AA
                email_input['required'] = True
                
                send_btn = soup.new_tag('button')
                send_btn['id'] = 'send-btn'
                send_btn['onclick'] = 'sendReport()'
                send_btn.string = 'Send Report'
                send_btn['style'] = 'padding: 12px 30px; font-size: 1.1rem; font-weight: bold; background: #E91E63; color: white; border: none; border-radius: 8px; cursor: pointer; transition: transform 0.3s, box-shadow 0.3s;'  # Darker pink for WCAG AA
                send_btn['onmouseover'] = "this.style.transform='scale(1.05)'; this.style.boxShadow='0 5px 20px rgba(233, 30, 99, 0.5)';"
                send_btn['onmouseout'] = "this.style.transform='scale(1)'; this.style.boxShadow='none';"
                
                status_div = soup.new_tag('div')
                status_div['id'] = 'email-status'
                status_div['style'] = 'text-align: center; min-height: 25px; font-weight: 600; color: #FF8FC7;'  # WCAG AA compliant
                
                email_form.append(email_input)
                email_form.append(send_btn)
                email_form.append(status_div)
                
                email_section.append(email_form)
                
                # Insert at the beginning of body
                body.insert(0, date_div)
                body.insert(1, email_section)
            
            return str(soup)
        except Exception as e:
            print(f"Warning: Could not add email form and date: {e}")
            return html_content
    
    def _save_html_output(self, result):
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
            
            # Ensure it's valid HTML
            if not html_content.strip().startswith("<!DOCTYPE") and not html_content.strip().startswith("<html"):
                # If the agent didn't provide full HTML, create a wrapper
                html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bitcoin Analysis Report - SEVENTEEN Edition 💎</title>
</head>
<body>
{html_content}
</body>
</html>"""
            
            # Add email form and date display
            html_content = self._add_email_form_and_date(html_content)
            
            # Save to index.html
            output_path = os.path.join(os.getcwd(), "index.html")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            print(f"\n✨ HTML report saved to: {output_path}")
            print(f"   Open it in your browser to see the SEVENTEEN-inspired report! Fighting! 💪")
            print(f"   Don't forget to start the email API: python3.11 email_api.py")
            
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

