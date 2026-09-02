import asyncio
import json
import re
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "qwen3:8b"

SAUCEDEMO_URL = "https://www.saucedemo.com/"
USERNAME = "standard_user"
PASSWORD = "secret_sauce"

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "saucedemo.json"

MAX_AGENT_STEPS = 30


# ============================================================
# PLAYWRIGHT MCP CLIENT
# ============================================================

client = MultiServerMCPClient(
    {
        "playwright": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "@playwright/mcp@latest",
                "--isolated",
            ],
        }
    }
)


# ============================================================
# LLM
# ============================================================

model = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)


# ============================================================
# DOM EXTRACTION JAVASCRIPT
# ============================================================

DOM_EXTRACTION_JS = r"""
() => {
    function clean(value) {
        if (value === null || value === undefined) {
            return null;
        }

        const result = String(value).trim();
        return result === "" ? null : result;
    }

    function cssEscape(value) {
        if (window.CSS && CSS.escape) {
            return CSS.escape(String(value));
        }

        return String(value).replace(
            /([ !"#$%&'()*+,./:;<=>?@[\\\]^`{|}~])/g,
            "\\$1"
        );
    }

    function cssSelector(el) {
        if (!(el instanceof Element)) {
            return null;
        }

        if (el.id) {
            return "#" + cssEscape(el.id);
        }

        const parts = [];
        let current = el;

        while (
            current &&
            current.nodeType === Node.ELEMENT_NODE &&
            current !== document.body &&
            current !== document.documentElement
        ) {
            let part = current.tagName.toLowerCase();

            const classes = Array.from(current.classList || [])
                .filter(Boolean)
                .slice(0, 3);

            if (classes.length > 0) {
                part += classes
                    .map(cls => "." + cssEscape(cls))
                    .join("");
            }

            const parent = current.parentElement;

            if (parent) {
                const sameTag = Array.from(parent.children)
                    .filter(child =>
                        child.tagName === current.tagName
                    );

                if (sameTag.length > 1) {
                    const index = sameTag.indexOf(current) + 1;
                    part += `:nth-of-type(${index})`;
                }
            }

            parts.unshift(part);
            current = parent;
        }

        return parts.join(" > ");
    }

    function xpathSelector(el) {
        if (!(el instanceof Element)) {
            return null;
        }

        const parts = [];
        let current = el;

        while (
            current &&
            current.nodeType === Node.ELEMENT_NODE
        ) {
            let index = 1;
            let sibling = current.previousElementSibling;

            while (sibling) {
                if (sibling.tagName === current.tagName) {
                    index++;
                }

                sibling = sibling.previousElementSibling;
            }

            parts.unshift(
                current.tagName.toLowerCase() + "[" + index + "]"
            );

            current = current.parentElement;
        }

        return "/" + parts.join("/");
    }

    function visible(el) {
        if (!(el instanceof Element)) {
            return false;
        }

        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();

        return (
            style.display !== "none" &&
            style.visibility !== "hidden" &&
            style.opacity !== "0" &&
            rect.width > 0 &&
            rect.height > 0
        );
    }

    function getText(el) {
        const ariaLabel = clean(
            el.getAttribute("aria-label")
        );

        if (ariaLabel) {
            return ariaLabel;
        }

        if (
            el.tagName === "INPUT" ||
            el.tagName === "TEXTAREA" ||
            el.tagName === "SELECT"
        ) {
            const value = clean(el.value);

            if (value) {
                return value;
            }
        }

        return clean(
            el.innerText || el.textContent
        );
    }

    function getType(el) {
        const tag = el.tagName.toLowerCase();

        if (tag === "input") {
            return (
                clean(el.getAttribute("type")) ||
                "text"
            );
        }

        return tag;
    }

    function extract(el) {
        return {
            type: getType(el),
            text: getText(el),
            id: clean(el.id),
            name: clean(el.getAttribute("name")),
            class: clean(el.getAttribute("class")),
            placeholder: clean(
                el.getAttribute("placeholder")
            ),
            aria_label: clean(
                el.getAttribute("aria-label")
            ),
            href: clean(
                el.getAttribute("href")
            ),
            css_selector: cssSelector(el),
            xpath: xpathSelector(el)
        };
    }

    const selector = [
        "button",
        "a",
        "input",
        "form",
        "textarea",
        "select",
        "input[type='checkbox']",
        "input[type='radio']"
    ].join(",");

    const elements = Array.from(
        document.querySelectorAll(selector)
    ).filter(visible);

    return {
        url: window.location.href,
        title: document.title,
        element_count: elements.length,
        elements: elements.map(extract)
    };
}
"""


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are a browser automation agent.

You control a real Playwright browser.

IMPORTANT:

Perform browser actions ONE AT A TIME.

Never issue multiple browser tool calls in a single response.

After each tool result, inspect the result before deciding
what the next browser action should be.

TASK:

1. Navigate to:
   {SAUCEDEMO_URL}

2. Inspect the page.

3. Locate the username input.

4. Locate the password input.

5. Enter:

   username: {USERNAME}
   password: {PASSWORD}

6. Click the Login button.

7. Wait for the login operation to complete.

8. Verify that login succeeded.

A successful login should show the SauceDemo inventory/products
page containing "Products" and "Swag Labs".

9. Once login is definitely successful, use browser_evaluate
to inspect the CURRENT DOM.

10. The DOM evaluation result is authoritative.

11. Do not invent selectors.

12. Do not return elements that are not currently visible.

13. Do not inspect or report elements from a previous page.

14. The final response must be JSON only.

15. Do not use markdown code fences.

16. The final JSON format must be:

{{
    "success": true,
    "url": "...",
    "title": "...",
    "elements": [
        {{
            "type": "...",
            "text": "...",
            "id": "...",
            "name": "...",
            "class": "...",
            "placeholder": "...",
            "aria_label": "...",
            "href": "...",
            "css_selector": "...",
            "xpath": "..."
        }}
    ]
}}

If login fails:

{{
    "success": false,
    "error": "...",
    "url": "...",
    "title": "...",
    "elements": []
}}

Never claim login succeeded unless the browser actually
shows the logged-in inventory/products page.
"""


# ============================================================
# HELPERS
# ============================================================

def clean_model_json(text: str) -> str:
    """
    Remove accidental markdown fences around JSON.
    """

    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


def parse_final_json(text: str) -> dict[str, Any]:
    """
    Parse model output as JSON.
    """

    text = clean_model_json(text)

    try:
        result = json.loads(text)

        if not isinstance(result, dict):
            raise ValueError(
                "Final JSON must be an object."
            )

        return result

    except json.JSONDecodeError:

        # Attempt recovery if the model added text
        # before/after the JSON object.

        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL
        )

        if not match:
            raise ValueError(
                "Model did not return valid JSON."
            )

        result = json.loads(match.group(0))

        if not isinstance(result, dict):
            raise ValueError(
                "Final JSON must be an object."
            )

        return result


def validate_result(data: dict[str, Any]) -> None:

    if "success" not in data:
        raise ValueError(
            "Missing 'success' field."
        )

    if data["success"] is False:
        return

    required = [
        "url",
        "title",
        "elements"
    ]

    for field in required:
        if field not in data:
            raise ValueError(
                f"Missing '{field}' field."
            )

    if not isinstance(
        data["elements"],
        list
    ):
        raise ValueError(
            "'elements' must be a list."
        )

    element_fields = [
        "type",
        "text",
        "id",
        "name",
        "class",
        "placeholder",
        "aria_label",
        "href",
        "css_selector",
        "xpath"
    ]

    for index, element in enumerate(
        data["elements"]
    ):

        if not isinstance(
            element,
            dict
        ):
            raise ValueError(
                f"Element {index} must be an object."
            )

        for field in element_fields:

            if field not in element:
                raise ValueError(
                    f"Element {index} missing "
                    f"'{field}'."
                )


def print_ai_step(
    response: AIMessage,
    step: int
):

    print("\n")
    print("=" * 80)
    print(f"AI STEP {step}")
    print("=" * 80)

    if response.content:
        print("\nCONTENT:")
        print(response.content)

    if response.tool_calls:

        print("\nNATIVE TOOL CALLS:")

        for call in response.tool_calls:

            print(
                f"  {call['name']}"
            )

            print(
                json.dumps(
                    call.get("args", {}),
                    indent=2,
                    ensure_ascii=False
                )
            )


def print_tool_result(
    result: ToolMessage,
    step: int
):

    print("\n")
    print("-" * 80)
    print(
        f"TOOL RESULT - STEP {step}"
    )
    print("-" * 80)

    content = result.content

    if isinstance(content, str):
        print(content[:15000])
    else:
        print(
            str(content)[:15000]
        )


# ============================================================
# MAIN
# ============================================================

async def main():

    print(
        "Starting Playwright MCP..."
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # DO NOT use:
    #
    #     tools = await client.get_tools()
    #
    # because that creates a new MCP session for each
    # tool invocation.
    #
    # Instead create one persistent session.
    # --------------------------------------------------------

    async with client.session(
        "playwright"
    ) as session:

        print(
            "Created persistent Playwright MCP session."
        )

        tools = await load_mcp_tools(
            session
        )

        print(
            f"Loaded {len(tools)} Playwright tools."
        )

        tool_names = {
            tool.name
            for tool in tools
        }

        print(
            "\nAvailable tools:"
        )

        for name in sorted(tool_names):
            print(
                f"  - {name}"
            )

        if "browser_evaluate" not in tool_names:
            raise RuntimeError(
                "browser_evaluate is not available."
            )

        # ----------------------------------------------------
        # Bind tools to Qwen.
        # ----------------------------------------------------

        model_with_tools = model.bind_tools(
            tools
        )

        messages = [
            HumanMessage(
                content=SYSTEM_PROMPT
                + """

Begin the browser task now.

Remember:
ONE browser tool call at a time.
"""
            )
        ]

        final_content = None

        # ----------------------------------------------------
        # Agent loop
        # ----------------------------------------------------

        for step in range(
            1,
            MAX_AGENT_STEPS + 1
        ):

            response = await (
                model_with_tools.ainvoke(
                    messages
                )
            )

            print_ai_step(
                response,
                step
            )

            messages.append(
                response
            )

            # ------------------------------------------------
            # Model has no tool call.
            # It should be producing final JSON.
            # ------------------------------------------------

            if not response.tool_calls:

                final_content = (
                    response.content
                )

                break

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Even if Qwen returns multiple tool calls,
            # execute ONLY THE FIRST ONE.
            # ------------------------------------------------

            call = response.tool_calls[0]

            tool_name = call["name"]

            tool_args = call.get(
                "args",
                {}
            )

            tool_call_id = call["id"]

            tool = next(
                (
                    t
                    for t in tools
                    if t.name == tool_name
                ),
                None
            )

            if tool is None:

                result = ToolMessage(
                    content=(
                        f"Unknown tool: "
                        f"{tool_name}"
                    ),
                    tool_call_id=tool_call_id
                )

                messages.append(
                    result
                )

                print_tool_result(
                    result,
                    step
                )

                continue

            print(
                "\nEXECUTING ONE TOOL:"
            )

            print(
                f"  {tool_name}"
            )

            print(
                json.dumps(
                    tool_args,
                    indent=2,
                    ensure_ascii=False
                )
            )

            # ------------------------------------------------
            # Execute inside the SAME persistent MCP session.
            # ------------------------------------------------

            try:

                tool_result = await tool.ainvoke(
                    tool_args
                )

                if isinstance(
                    tool_result,
                    str
                ):
                    content = tool_result
                else:
                    content = json.dumps(
                        tool_result,
                        ensure_ascii=False,
                        default=str
                    )

            except Exception as exc:

                content = (
                    f"Tool execution error: "
                    f"{type(exc).__name__}: {exc}"
                )

            result = ToolMessage(
                content=content,
                tool_call_id=tool_call_id
            )

            print_tool_result(
                result,
                step
            )

            messages.append(
                result
            )

        # ----------------------------------------------------
        # Agent exceeded maximum steps.
        # ----------------------------------------------------

        if final_content is None:

            raise RuntimeError(
                f"Agent exceeded "
                f"{MAX_AGENT_STEPS} steps "
                f"without returning JSON."
            )

        # ----------------------------------------------------
        # Parse final JSON.
        # ----------------------------------------------------

        print("\n")
        print("=" * 80)
        print("RAW FINAL RESPONSE")
        print("=" * 80)

        print(
            final_content
        )

        if isinstance(
            final_content,
            list
        ):

            final_content = "".join(
                item.get(
                    "text",
                    ""
                )
                if isinstance(
                    item,
                    dict
                )
                else str(item)
                for item in final_content
            )

        data = parse_final_json(
            final_content
        )

        validate_result(
            data
        )

        # ----------------------------------------------------
        # Save.
        # ----------------------------------------------------

        with OUTPUT_FILE.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )

        print("\n")
        print("=" * 80)
        print("SUCCESS")
        print("=" * 80)

        print(
            f"Output: "
            f"{OUTPUT_FILE.resolve()}"
        )

        print(
            f"Login success: "
            f"{data.get('success')}"
        )

        print(
            f"Elements: "
            f"{len(data.get('elements', []))}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())