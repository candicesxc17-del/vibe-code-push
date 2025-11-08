"""
API endpoint for sending Bitcoin analysis reports via email
"""

import os
import smtplib
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the HTML page

def extract_text_from_html(html_file):
    """Extract and clean text content from HTML file"""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

def format_professional_report(raw_text):
    """Format the extracted text in a professional Wall Street trader tone"""
    
    text = raw_text
    
    # Replace common Gen Z phrases with professional equivalents
    replacements = {
        'no cap': '',
        'periodt': '',
        'vibe check': 'Market Assessment',
        'bestie': '',
        'lowkey': '',
        'highkey': '',
        'fr fr': '',
        'that\'s facts': '',
        'stay woke': '',
        'it\'s giving': 'indicating',
        'slay': 'perform well',
        'fire': 'strong',
        'lit': 'active',
        'tea': 'information',
        'spill the tea': 'provide details',
    }
    
    for slang, professional in replacements.items():
        text = re.sub(rf'\b{re.escape(slang)}\b', professional, text, flags=re.IGNORECASE)
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    return text

@app.route('/send-report', methods=['POST'])
def send_report():
    """API endpoint to send Bitcoin analysis report via email"""
    try:
        data = request.json
        recipient_email = data.get('email')
        
        if not recipient_email:
            return jsonify({'success': False, 'error': 'Email address is required'}), 400
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, recipient_email):
            return jsonify({'success': False, 'error': 'Invalid email format'}), 400
        
        # Get credentials from .env
        gmail_email = os.getenv('GMAIL_EMAIL')
        gmail_password = os.getenv('GMAIL_PASSWORD')
        
        if not gmail_email or not gmail_password:
            return jsonify({'success': False, 'error': 'Email service not configured'}), 500
        
        # Check if index.html exists
        html_file = 'index.html'
        if not os.path.exists(html_file):
            return jsonify({'success': False, 'error': 'Report file not found'}), 404
        
        # Extract and format text from HTML
        raw_text = extract_text_from_html(html_file)
        if not raw_text:
            return jsonify({'success': False, 'error': 'Failed to extract report content'}), 500
        
        professional_report = format_professional_report(raw_text)
        
        # Create email
        msg = MIMEMultipart()
        msg['From'] = gmail_email
        msg['To'] = recipient_email
        msg['Subject'] = 'Bitcoin Trading Analysis Report - Market Intelligence'
        
        # Email body
        body = f"""BITCOIN MARKET ANALYSIS REPORT
{'-' * 50}

Dear Valued Client,

Please find below our comprehensive Bitcoin market analysis and trading recommendations based on recent market data and sentiment analysis.

{professional_report}

{'-' * 50}

TRADING RECOMMENDATION SUMMARY

Based on our analysis of current market conditions, technical indicators, and fundamental factors, we provide the following actionable insights for your consideration.

Please note: This analysis is for informational purposes only and should not be considered as financial advice. Always conduct your own due diligence and consult with a qualified financial advisor before making trading decisions.

Best regards,
Bitcoin Analysis System

---
This report was generated automatically based on real-time market data and sentiment analysis.
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_email, gmail_password)
        server.sendmail(gmail_email, recipient_email, msg.as_string())
        server.quit()
        
        return jsonify({'success': True, 'message': f'Report sent successfully to {recipient_email}'}), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    port = int(os.getenv('EMAIL_API_PORT', '5050'))
    app.run(host='0.0.0.0', port=port, debug=True)

