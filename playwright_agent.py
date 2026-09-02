import asyncio
import json
import os
import re

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_ollama import ChatOllama


async def main():

    
    # 1. Initialize Ollama model

    model = ChatOllama(
        model="qwen2.5-coder:7b",
        temperature=0
    )

    
    # 2. Connect to Playwright MCP server
    client = MultiServerMCPClient({
        "playwright": {
        "transport": "stdio",
        "command": "npx",
        "args": ["@playwright/mcp@latest"],
    }
})
    
    # 3. Get Playwright tools
    
    tools = await client.get_tools()

    print("Available Playwright tools:")
    for tool in tools:
        print("-", tool.name)

   
    # 4. Create agent
   
    agent = create_agent(
        model,
        tools
    )

    # 5. User request

    user_request = """
    open browser and go to https://www.saucedemo.com/?utm_source=chatgpt.com, give crendentials Username: standard_user
Password: secret_sauce, do login and bring all pageobjects in the home page

    Inspect the webpage using Playwright.

    Identify all important page objects, especially:
    - buttons
    - links
    - input fields
    - forms
    - checkboxes
    - radio buttons
    - select elements
    - textareas

    For every page object identify:
    - type
    - visible text
    - id
    - name
    - class
    - placeholder
    - aria-label
    - href
    - CSS selector
    - XPath

    Save this information with the name "example_page".
    """

    # 6. Agent prompt

    prompt = f"""
You are a browser automation agent.

USER REQUEST:
{user_request}

IMPORTANT INSTRUCTIONS:

1. Use the Playwright tools to open the webpage.
2. Inspect the REAL DOM.
3. Do not guess element properties.
4. Identify useful interactive page objects.
5. Generate reliable CSS selectors.
6. Generate XPath where appropriate.
7. Return ONLY valid JSON.
8. Do NOT use Markdown code fences.
9. Do NOT include explanations outside the JSON.

Return JSON in exactly this structure:

{{
    "name": "example_page",
    "url": "",
    "title": "",
    "page_objects": [
        {{
            "type": "",
            "text": "",
            "id": "",
            "name": "",
            "class": "",
            "placeholder": "",
            "aria_label": "",
            "href": "",
            "css_selector": "",
            "xpath": ""
        }}
    ]
}}
"""

    # 7. Run the agent

    print("\nStarting browser agent...\n")

    response = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

   
    # 8. Get final agent response
  
    final_message = response["messages"][-1]

    json_text = final_message.content

    print("\nAgent response:")
    print(json_text)

    # 9. Remove Markdown code fences

    json_text = re.sub(
        r"```json\s*|\s*```",
        "",
        json_text
    ).strip()


    # 10. Parse JSON
  
    try:
        page_data = json.loads(json_text)

    except json.JSONDecodeError as e:

        print("\nERROR: Model did not return valid JSON.")
        print("JSON error:", e)
        print("\nRaw response:")
        print(json_text)

        return

   
    # 11. Create data directory
   
    os.makedirs("data", exist_ok=True)

   
    # 12. Get JSON filename

    name = page_data.get(
        "name",
        "page_objects"
    )

    # Make filename safe
    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        name
    )

    filename = f"data/{safe_name}.json"

 
    # 13. Save JSON
    # ----------------------------------------
    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            page_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    
    # 14. Done
   
    print("\n--------------------------------")
    print("SUCCESS")
    print("--------------------------------")
    print(f"JSON saved to: {filename}")
    print("--------------------------------")


if __name__ == "__main__":
    asyncio.run(main())