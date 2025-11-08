# Email API Setup

## How to Use the Email Functionality

The `index.html` page includes an email form that allows users to send the Bitcoin analysis report to their email address.

### 1. Start the Email API Server

Before users can send emails, you need to start the email API server:

```bash
python3.11 email_api.py
```

The server will run on `http://localhost:5050` by default. To use a different port, set `EMAIL_API_PORT` in your environment before starting the server.

### 2. Open index.html

Open `index.html` in your browser. You'll see:
- Current date displayed at the top (updates automatically)
- Email form with input box and "Send Report" button

### 3. Send Report via Email

1. Enter an email address in the text box
2. Click "Send Report"
3. The report will be sent to that email address
4. You'll see a success or error message

### Requirements

- Gmail credentials must be set in `.env` file:
  - `GMAIL_EMAIL=your_email@gmail.com`
  - `GMAIL_PASSWORD=your_app_password`

### API Endpoints

- `POST /send-report` - Send report to email
  - Body: `{"email": "user@example.com"}`
  - Returns: `{"success": true, "message": "..."}` or error

- `GET /health` - Health check endpoint

### Notes

- The email API must be running for the email form to work
- The date on the page updates automatically using JavaScript
- Email validation is performed on both client and server side
