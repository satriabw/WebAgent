import asyncio

from playwright.async_api import async_playwright
from agents.webvoyager import WebVoyager

# Example usage
async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://www.google.com")

        question = "Could you explain the Fermat Last Theorem (Wikipedia)?"
        final_response = await WebVoyager().call_agent(question, page)
        print(f"Final response: {final_response}")

if __name__ == "__main__":
    asyncio.run(main())
