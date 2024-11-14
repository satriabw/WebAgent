# WebAgent 🤖

WebAgent is an intelligent web automation tool that helps you automate browser-based tasks using OpenAI's API. Modification from https://langchain-ai.github.io/langgraph/tutorials/web-navigation/web_voyager/#setup

## 🌟 Features

- Browser automation with advanced AI capabilities
- Natural language processing for web interactions
- Customizable automation workflows
- Multi-browser support
- Detailed logging and reporting

## 📋 Prerequisites

- Python 3.8 or higher
- Chrome or Firefox browser
- OpenAI API key

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/satriabw/WebAgent.git
cd WebAgent
```

2. Install required dependencies:
```bash
conda env create -f requirements.yml
```

3. Set up your OpenAI API key:
```bash
# Linux/macOS
export OPENAI_SECRET_KEY=your_api_key_here

# Windows (Command Prompt)
set OPENAI_SECRET_KEY=your_api_key_here

# Windows (PowerShell)
$env:OPENAI_SECRET_KEY="your_api_key_here"
```

or create `.env` file in the folder and set
```bash
OPENAI_SECRET_KEY=your_api_key_here
```

## 🚀 Quick Start
1. Open main.py, change `question` variable to the task you wanted to perform.

2. Run the main application:
```bash
python main.py
```


