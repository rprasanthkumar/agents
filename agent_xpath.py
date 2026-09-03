import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
JSON_DIR = DATA_DIR / "pages"
LOG_DIR = DATA_DIR / "logs"

OLLAMA_MODEL = "qwen3:8b"

DEFAULT_URL = "https://www.saucedemo.com/"

# Playwright MCP is started with an isolated browser profile.
# A persistent MCP session is used so browser state is preserved
# across navigation, login, clicks, snapshots, and extraction.

MCP_CONFIG = {
    "playwright": {
        "transport": "stdio",
        "command": "npx",
        "args": [
            "@playwright/mcp@0.0.80",
            "--isolated",
        ],
    }
}


st.set_page_config(
    page_title="Browser Automation Agent",
    layout="wide",
)


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def create_directories():
    """Create directories used for generated files."""

    SNAPSHOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# MCP TOOL HELPERS
# ============================================================

def get_tool(tools, name: str):
    """Return an MCP tool by name."""

    for tool in tools:
        if tool.name == name:
            return tool

    available_tools = [
        tool.name
        for tool in tools
    ]

    raise RuntimeError(
        f"Tool '{name}' was not found. "
        f"Available tools: {available_tools}"
    )


async def execute_tool(
    tool,
    arguments: dict,
    activity: list,
):
    """
    Execute an MCP tool and record its status.
    """

    activity.append(
        {
            "tool": tool.name,
            "status": "running",
        }
    )

    try:
        result = await tool.ainvoke(arguments)

        activity[-1]["status"] = "success"

        return result

    except Exception as exc:

        activity[-1]["status"] = "failed"

        activity[-1]["error"] = str(exc)

        raise


# ============================================================
# MCP RESULT EXTRACTION
# ============================================================

def extract_text(result: Any) -> str:
    """
    Extract text from an MCP/LangChain result.

    MCP responses can be represented as strings, dictionaries,
    lists of content blocks, or objects containing text/content.
    """

    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):

        if isinstance(
            result.get("text"),
            str,
        ):
            return result["text"]

        if "content" in result:
            return extract_text(
                result["content"]
            )

        if "result" in result:
            return extract_text(
                result["result"]
            )

        if "value" in result:
            return extract_text(
                result["value"]
            )

        return json.dumps(
            result,
            ensure_ascii=False,
        )

    if isinstance(result, list):

        parts = []

        for item in result:

            if isinstance(item, str):

                parts.append(item)

            elif isinstance(item, dict):

                if isinstance(
                    item.get("text"),
                    str,
                ):
                    parts.append(
                        item["text"]
                    )

                elif "content" in item:

                    parts.append(
                        extract_text(
                            item["content"]
                        )
                    )

                else:

                    parts.append(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                        )
                    )

            elif hasattr(item, "text"):

                parts.append(
                    str(item.text)
                )

            elif hasattr(item, "content"):

                parts.append(
                    extract_text(
                        item.content
                    )
                )

            else:

                parts.append(
                    str(item)
                )

        return "\n".join(parts)

    if hasattr(result, "text"):
        return str(result.text)

    if hasattr(result, "content"):
        return extract_text(
            result.content
        )

    return str(result)


def parse_json_result(result: Any):
    """
    Parse JSON returned by browser_evaluate.

    The JSON may be returned directly or wrapped inside
    an MCP text/content response.
    """

    if isinstance(result, dict):

        if "result" in result:

            value = result["result"]

            if isinstance(
                value,
                (dict, list),
            ):
                return value

        if "value" in result:

            value = result["value"]

            if isinstance(
                value,
                (dict, list),
            ):
                return value

    text = extract_text(result).strip()

    if not text:
        raise ValueError(
            "browser_evaluate returned an empty result."
        )

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # Some MCP responses may contain additional text around
    # the JSON value. Try to locate the first valid JSON object
    # or array.

    decoder = json.JSONDecoder()

    for index, character in enumerate(text):

        if character not in "[{":
            continue

        try:

            value, _ = decoder.raw_decode(
                text[index:]
            )

            return value

        except json.JSONDecodeError:
            continue

    raise ValueError(
        "Could not parse JSON from browser_evaluate:\n"
        + text[:3000]
    )


def extract_snapshot_text(result: Any) -> str:
    """
    Extract the raw browser_snapshot text.

    The snapshot is deliberately not parsed or reconstructed.
    This preserves the format produced by Playwright MCP.
    """

    text = extract_text(result).strip()

    if not text:

        raise ValueError(
            "browser_snapshot returned an empty snapshot."
        )

    return text


# ============================================================
# FILE NAME HELPER
# ============================================================

def safe_filename(value: str) -> str:
    """Convert a URL into a safe filename."""

    value = value.strip()

    value = re.sub(
        r"https?://",
        "",
        value,
    )

    value = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        value,
    )

    value = value.strip("_")

    if not value:
        value = "page"

    return value[:100]


# ============================================================
# DOM EXTRACTION JAVASCRIPT
# ============================================================

DOM_EXTRACTION_JS = r"""
() => {

    const elements = Array.from(
        document.querySelectorAll(
            "button, a, input, form, textarea, select, [role]"
        )
    );


    // ========================================================
    // VISIBILITY
    // ========================================================

    function isVisible(element) {

        const style =
            window.getComputedStyle(element);

        const rect =
            element.getBoundingClientRect();

        if (style.display === "none") {
            return false;
        }

        if (style.visibility === "hidden") {
            return false;
        }

        if (style.opacity === "0") {
            return false;
        }

        if (element.hidden) {
            return false;
        }

        if (
            element.getAttribute("aria-hidden") === "true"
        ) {
            return false;
        }

        return (
            rect.width > 0 &&
            rect.height > 0
        );
    }


    // ========================================================
    // TEXT
    // ========================================================

    function cleanText(element) {

        return (
            element.innerText ||
            element.textContent ||
            ""
        )
            .replace(/\s+/g, " ")
            .trim();
    }


    // ========================================================
    // ARIA LABEL
    // ========================================================

    function getAriaLabel(element) {

        const directLabel =
            element.getAttribute("aria-label");

        if (directLabel) {
            return directLabel.trim();
        }

        const labelledBy =
            element.getAttribute("aria-labelledby");

        if (!labelledBy) {
            return "";
        }

        return labelledBy
            .split(/\s+/)
            .map(
                id =>
                    document.getElementById(id)
            )
            .filter(Boolean)
            .map(
                element =>
                    cleanText(element)
            )
            .filter(Boolean)
            .join(" ");
    }


    // ========================================================
    // ELEMENT TYPE
    // ========================================================

    function getType(element) {

        const tag =
            element.tagName.toLowerCase();

        if (tag !== "input") {
            return tag;
        }

        const inputType = (
            element.getAttribute("type") ||
            "text"
        ).toLowerCase();

        if (
            inputType === "checkbox" ||
            inputType === "radio"
        ) {
            return inputType;
        }

        return "input";
    }


    // ========================================================
    // CSS ESCAPE
    // ========================================================

    function cssEscape(value) {

        if (
            window.CSS &&
            typeof window.CSS.escape === "function"
        ) {
            return window.CSS.escape(value);
        }

        return String(value).replace(
            /([!"#$%&'()*+,./:;<=>?@[\\\]^`{|}~])/g,
            "\\$1"
        );
    }


    // ========================================================
    // CSS SELECTOR
    // ========================================================

    function getCssSelector(element) {

        if (element.id) {

            return (
                "#" +
                cssEscape(element.id)
            );
        }

        const parts = [];

        let current = element;

        while (
            current &&
            current.nodeType === Node.ELEMENT_NODE
        ) {

            let selector =
                current.tagName.toLowerCase();

            if (current.id) {

                selector +=
                    "#" +
                    cssEscape(current.id);

                parts.unshift(selector);

                break;
            }

            const parent =
                current.parentElement;

            if (!parent) {

                parts.unshift(selector);

                break;
            }

            const siblings =
                Array.from(
                    parent.children
                ).filter(
                    sibling =>
                        sibling.tagName ===
                        current.tagName
                );

            if (siblings.length > 1) {

                const index =
                    siblings.indexOf(
                        current
                    ) + 1;

                selector +=
                    `:nth-of-type(${index})`;
            }

            parts.unshift(selector);

            current = parent;
        }

        return parts.join(" > ");
    }


    // ========================================================
    // ABSOLUTE XPATH
    // ========================================================

    function getAbsoluteXPath(element) {

        /*
         * Keep the absolute XPath genuinely absolute.
         *
         * Example:
         *
         * /html[1]/body[1]/div[1]
         *
         * Unlike the relative XPath below, this represents
         * the complete DOM hierarchy.
         */

        const parts = [];

        let current = element;

        while (
            current &&
            current.nodeType === Node.ELEMENT_NODE
        ) {

            let index = 1;

            let sibling =
                current.previousElementSibling;

            while (sibling) {

                if (
                    sibling.tagName ===
                    current.tagName
                ) {
                    index++;
                }

                sibling =
                    sibling.previousElementSibling;
            }

            parts.unshift(
                current.tagName.toLowerCase() +
                `[${index}]`
            );

            current =
                current.parentElement;
        }

        return "/" + parts.join("/");
    }


    // ========================================================
    // XPATH STRING LITERAL
    // ========================================================

    function xpathLiteral(value) {

        value = String(value);

        /*
         * XPath string with no single quote.
         */
        if (!value.includes("'")) {

            return "'" + value + "'";
        }

        /*
         * XPath string with no double quote.
         */
        if (!value.includes('"')) {

            return '"' + value + '"';
        }

        /*
         * String contains both single and double quotes.
         * XPath requires concat().
         */

        const parts =
            value.split("'");

        let result = "concat(";

        parts.forEach(
            (part, index) => {

                if (index > 0) {
                    result += ', "\'", ';
                }

                result += "'" + part + "'";
            }
        );

        result += ")";

        return result;
    }


    // ========================================================
    // CHECK XPATH UNIQUENESS
    // ========================================================

    function isUniqueXPath(xpath, element) {

        try {

            const result =
                document.evaluate(
                    xpath,
                    document,
                    null,
                    XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,
                    null
                );

            return (
                result.snapshotLength === 1 &&
                result.snapshotItem(0) === element
            );

        } catch (error) {

            return false;
        }
    }


    // ========================================================
    // RELATIVE XPATH
    // ========================================================

    function getRelativeXPath(element) {

        const tag =
            element.tagName.toLowerCase();

        /*
         * ----------------------------------------------------
         * 1. ID
         * ----------------------------------------------------
         */

        if (element.id) {

            const xpath =
                `//*[@id=${xpathLiteral(element.id)}]`;

            if (
                isUniqueXPath(
                    xpath,
                    element
                )
            ) {
                return xpath;
            }
        }


        /*
         * ----------------------------------------------------
         * 2. NAME
         * ----------------------------------------------------
         */

        const name =
            element.getAttribute("name");

        if (name) {

            const xpath =
                `//${tag}[@name=${xpathLiteral(name)}]`;

            if (
                isUniqueXPath(
                    xpath,
                    element
                )
            ) {
                return xpath;
            }
        }


        /*
         * ----------------------------------------------------
         * 3. ARIA LABEL
         * ----------------------------------------------------
         */

        const ariaLabel =
            element.getAttribute("aria-label");

        if (ariaLabel) {

            const xpath =
                `//${tag}[@aria-label=${xpathLiteral(
                    ariaLabel.trim()
                )}]`;

            if (
                isUniqueXPath(
                    xpath,
                    element
                )
            ) {
                return xpath;
            }
        }


        /*
         * ----------------------------------------------------
         * 4. PLACEHOLDER
         * ----------------------------------------------------
         */

        const placeholder =
            element.getAttribute("placeholder");

        if (placeholder) {

            const xpath =
                `//${tag}[@placeholder=${xpathLiteral(
                    placeholder
                )}]`;

            if (
                isUniqueXPath(
                    xpath,
                    element
                )
            ) {
                return xpath;
            }
        }


        /*
         * ----------------------------------------------------
         * 5. DATA-TEST / DATA-TESTID
         * ----------------------------------------------------
         */

        const testId =
            element.getAttribute("data-testid") ||
            element.getAttribute("data-test") ||
            element.getAttribute("data-cy");

        if (testId) {

            let attribute = "data-testid";

            if (
                element.hasAttribute("data-test")
            ) {
                attribute = "data-test";
            }

            if (
                element.hasAttribute("data-cy")
            ) {
                attribute = "data-cy";
            }

            const xpath =
                `//${tag}[@${attribute}=${xpathLiteral(
                    testId
                )}]`;

            if (
                isUniqueXPath(
                    xpath,
                    element
                )
            ) {
                return xpath;
            }
        }


        /*
         * ----------------------------------------------------
         * 6. ROLE + TEXT
         * ----------------------------------------------------
         */

        const role =
            element.getAttribute("role");

        const text = cleanText(element);

        if (
            role &&
            text
        ) {

            const xpath =
                `//${tag}[@role=${xpathLiteral(
                    role
                )} and normalize-space()=${xpathLiteral(
                    text
                )}]`;

            if (
                isUniqueXPath(
                    xpath,
                    element
                )
            ) {
                return xpath;
            }
        }


        /*
         * ----------------------------------------------------
         * 7. BUTTON / LINK TEXT
         * ----------------------------------------------------
         */

        if (
            (
                tag === "button" ||
                tag === "a"
            ) &&
            text
        ) {

            const xpath =
                `//${tag}[normalize-space()=${xpathLiteral(
                    text
                )}]`;

            if (
                isUniqueXPath(
                    xpath,
                    element
                )
            ) {
                return xpath;
            }
        }


        /*
         * ----------------------------------------------------
         * 8. INPUT TYPE
         * ----------------------------------------------------
         */

        if (tag === "input") {

            const inputType =
                element.getAttribute("type");

            if (inputType) {

                const xpath =
                    `//input[@type=${xpathLiteral(
                        inputType
                    )}]`;

                if (
                    isUniqueXPath(
                        xpath,
                        element
                    )
                ) {
                    return xpath;
                }
            }
        }


        /*
         * ----------------------------------------------------
         * 9. UNIQUE CLASS
         * ----------------------------------------------------
         */

        const className =
            element.getAttribute("class") || "";

        const classes =
            className
                .split(/\s+/)
                .filter(Boolean);

        if (classes.length > 0) {

            /*
             * First try individual classes.
             */

            for (const cls of classes) {

                const xpath =
                    `//${tag}[contains(` +
                    `concat(' ', normalize-space(@class), ' '), ` +
                    `${xpathLiteral(" " + cls + " ")}` +
                    `)]`;

                if (
                    isUniqueXPath(
                        xpath,
                        element
                    )
                ) {
                    return xpath;
                }
            }


            /*
             * Then try the complete class combination.
             */

            const conditions =
                classes
                    .map(
                        cls =>
                            `contains(` +
                            `concat(' ', normalize-space(@class), ' '), ` +
                            `${xpathLiteral(
                                " " + cls + " "
                            )})`
                    )
                    .join(" and ");

            const xpath =
                `//${tag}[${conditions}]`;

            if (
                isUniqueXPath(
                    xpath,
                    element
                )
            ) {
                return xpath;
            }
        }


        /*
         * ----------------------------------------------------
         * 10. UNIQUE TAG
         * ----------------------------------------------------
         */

        const tagXPath =
            `//${tag}`;

        if (
            isUniqueXPath(
                tagXPath,
                element
            )
        ) {
            return tagXPath;
        }


        /*
         * ----------------------------------------------------
         * 11. RELATIVE STRUCTURAL XPATH
         * ----------------------------------------------------
         *
         * Example:
         *
         * //div[2]/form[1]/input[3]
         *
         * This is still relative because it starts with //.
         */

        const parts = [];

        let current = element;

        while (
            current &&
            current.nodeType === Node.ELEMENT_NODE
        ) {

            const currentTag =
                current.tagName.toLowerCase();

            let index = 1;

            let sibling =
                current.previousElementSibling;

            while (sibling) {

                if (
                    sibling.tagName ===
                    current.tagName
                ) {
                    index++;
                }

                sibling =
                    sibling.previousElementSibling;
            }

            parts.unshift(
                `${currentTag}[${index}]`
            );

            /*
             * If a parent has an ID, use it as the
             * starting point instead of continuing
             * all the way to html/body.
             */

            if (current.parentElement) {

                const parent =
                    current.parentElement;

                if (parent.id) {

                    parts.unshift(
                        `*[@id=${xpathLiteral(
                            parent.id
                        )}]`
                    );

                    return (
                        "//" +
                        parts.join("/")
                    );
                }
            }

            current =
                current.parentElement;
        }

        return (
            "//" +
            parts.join("/")
        );
    }


    // ========================================================
    // EXTRACT ELEMENTS
    // ========================================================

    const result = [];

    for (const element of elements) {

        if (!isVisible(element)) {
            continue;
        }

        const tag =
            element.tagName.toLowerCase();

        result.push({

            type:
                getType(element),

            role:
                element.getAttribute("role") || "",

            text:
                cleanText(element),

            id:
                element.id || "",

            name:
                element.getAttribute("name") || "",

            class:
                element.getAttribute("class") || "",

            placeholder:
                element.getAttribute(
                    "placeholder"
                ) || "",

            aria_label:
                getAriaLabel(element),

            href:
                tag === "a"
                    ? element.href || ""
                    : "",

            css_selector:
                getCssSelector(element),

            /*
             * Absolute XPath.
             */
            xpath:
                getAbsoluteXPath(element),

            /*
             * Relative XPath.
             */
            relative_xpath:
                getRelativeXPath(element),
        });
    }


    // ========================================================
    // FINAL DOM OBJECT
    // ========================================================

    return {

        url:
            window.location.href,

        title:
            document.title,

        element_count:
            result.length,

        elements:
            result,
    };
}
"""


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt(
    username: str,
    password_supplied: bool,
) -> str:
    """
    Build the instructions given to Qwen.

    Qwen controls the browser through MCP.
    Python is responsible for extracting and saving
    the final JSON and raw Playwright snapshot.
    """

    username_instruction = ""

    if username:

        username_instruction = (
            f"The username supplied by the user is: {username}"
        )

    password_instruction = ""

    if password_supplied:

        password_instruction = (
            "A password was supplied through the UI. "
            "Use it when the user explicitly asks you to log in. "
            "Never print the password in your response."
        )

    return f"""
You are a browser automation agent.

You have access to Playwright MCP browser tools.

Execute the user's browser task using the available tools.

{username_instruction}

{password_instruction}

Browser automation rules:

1. Use actual Playwright tools to perform the requested actions.

2. When you encounter an unfamiliar page, use browser_snapshot
   to inspect the page before interacting with it.

3. Prefer accessible roles, names, labels and stable IDs.

4. Do not invent elements or selectors.

5. If the user requests login, actually perform the login.

6. If the login fails, inspect the page and determine why.

7. If navigation leads to another page, inspect the new page.

8. Complete the requested workflow before responding.

9. Do not fabricate JSON or YAML.

10. Python will capture the real browser_snapshot and the DOM
    after your browser workflow finishes.

11. Never expose passwords in the final response.

12. When the user asks for page objects, inspect the relevant
    page and leave the browser on the final relevant page so
    the extraction layer can capture it.

13. Do not stop the workflow merely because a cookie banner,
    dialog, or popup exists. Inspect it and handle it when
    necessary for the requested task.
"""


# ============================================================
# BROWSER AGENT
# ============================================================

async def run_browser_agent(
    instruction: str,
    url: str,
    username: str,
    password: str,
):
    """
    Start a persistent Playwright MCP session,
    let Qwen execute the browser task, then capture
    the final page as YAML and JSON.
    """

    create_directories()

    activity = []

    client = MultiServerMCPClient(
        MCP_CONFIG
    )

    # Keep one MCP session alive for the complete workflow.
    # Browser state therefore persists between:
    #
    # navigation
    # login
    # clicks
    # snapshots
    # extraction

    async with client.session(
        "playwright"
    ) as session:

        tools = await load_mcp_tools(
            session
        )

        activity.append(
            {
                "tool": "MCP",
                "status": "connected",
                "tool_count": len(tools),
            }
        )

        # ----------------------------------------------------
        # QWEN MODEL
        # ----------------------------------------------------

        model = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
        )

        system_prompt = build_system_prompt(
            username=username,
            password_supplied=bool(password),
        )

        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
        )

        user_prompt = f"""
Execute this browser task:

{instruction}

Starting URL:

{url}
"""

        final_state = None

        # ----------------------------------------------------
        # RUN AGENT
        # ----------------------------------------------------

        async for update in agent.astream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ]
            },
            stream_mode="updates",
        ):

            final_state = update

            if not isinstance(
                update,
                dict,
            ):
                continue

            for node_data in update.values():

                if not isinstance(
                    node_data,
                    dict,
                ):
                    continue

                messages = node_data.get(
                    "messages"
                )

                if not messages:
                    continue

                for message in messages:

                    message_type = getattr(
                        message,
                        "type",
                        "",
                    )

                    if message_type != "tool":
                        continue

                    tool_name = getattr(
                        message,
                        "name",
                        "unknown",
                    )

                    activity.append(
                        {
                            "tool": tool_name,
                            "status": "completed",
                        }
                    )


        # ====================================================
        # CAPTURE PLAYWRIGHT SNAPSHOT
        # ====================================================

        browser_snapshot = get_tool(
            tools,
            "browser_snapshot",
        )

        snapshot_result = await execute_tool(
            browser_snapshot,
            {},
            activity,
        )

        snapshot_text = extract_snapshot_text(
            snapshot_result
        )


        # ====================================================
        # CAPTURE DOM
        # ====================================================

        browser_evaluate = get_tool(
            tools,
            "browser_evaluate",
        )

        dom_result = await execute_tool(
            browser_evaluate,
            {
                "function": DOM_EXTRACTION_JS,
            },
            activity,
        )

        dom_data = parse_json_result(
            dom_result
        )

        if not isinstance(
            dom_data,
            dict,
        ):

            raise ValueError(
                "DOM extraction did not return an object."
            )


        # ====================================================
        # NORMALIZE ELEMENT COUNT
        # ====================================================

        elements = dom_data.get(
            "elements",
            [],
        )

        if not isinstance(
            elements,
            list,
        ):
            elements = []

        dom_data["element_count"] = len(
            elements
        )


        # ====================================================
        # FILE NAMES
        # ====================================================

        current_url = str(
            dom_data.get(
                "url",
                url,
            )
        )

        page_name = safe_filename(
            current_url
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        snapshot_path = (
            SNAPSHOT_DIR /
            f"{page_name}_{timestamp}.yaml"
        )

        json_path = (
            JSON_DIR /
            f"{page_name}_{timestamp}.json"
        )

        log_path = (
            LOG_DIR /
            f"playwright_{timestamp}.json"
        )


        # ====================================================
        # SAVE RAW PLAYWRIGHT SNAPSHOT
        # ====================================================

        snapshot_path.write_text(
            snapshot_text + "\n",
            encoding="utf-8",
        )


        # ====================================================
        # SAVE DOM JSON
        # ====================================================

        json_path.write_text(
            json.dumps(
                dom_data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


        # ====================================================
        # SAVE TOOL LOG
        # ====================================================

        log_path.write_text(
            json.dumps(
                activity,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


        # ====================================================
        # GET FINAL AI RESPONSE
        # ====================================================

        agent_response = ""

        if isinstance(
            final_state,
            dict,
        ):

            for node_data in final_state.values():

                if not isinstance(
                    node_data,
                    dict,
                ):
                    continue

                messages = node_data.get(
                    "messages"
                )

                if not messages:
                    continue

                for message in reversed(
                    messages
                ):

                    message_type = getattr(
                        message,
                        "type",
                        "",
                    )

                    if message_type != "ai":
                        continue

                    content = getattr(
                        message,
                        "content",
                        "",
                    )

                    if isinstance(
                        content,
                        str,
                    ):

                        agent_response = content

                    elif isinstance(
                        content,
                        list,
                    ):

                        agent_response = "\n".join(
                            str(item)
                            for item in content
                        )

                    break


        # ====================================================
        # RETURN EVERYTHING
        # ====================================================

        return {

            "agent_response":
                agent_response,

            "url":
                current_url,

            "snapshot":
                snapshot_text,

            "dom":
                dom_data,

            "snapshot_path":
                snapshot_path,

            "json_path":
                json_path,

            "log_path":
                log_path,

            "activity":
                activity,
        }


# ============================================================
# STREAMLIT ACTIVITY
# ============================================================

def render_activity(activity):

    """
    Display tool activity in Streamlit.
    Icons are intentionally not used.
    """

    for item in activity:

        tool_name = item.get(
            "tool",
            "unknown",
        )

        status = item.get(
            "status",
            "unknown",
        )

        st.write(
            f"`{tool_name}` — {status}"
        )

        if item.get("error"):

            st.error(
                item["error"]
            )


# ============================================================
# STREAMLIT MAIN
# ============================================================

def main():

    """Run the Streamlit interface."""

    st.title(
        "Browser Automation Agent"
    )

    st.write(
        "Enter a browser task in natural language. "
        "Qwen will control Playwright and the application "
        "will save the resulting page snapshot and DOM data."
    )


    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.header(
            "Browser"
        )

        url = st.text_input(
            "Starting URL",
            value=DEFAULT_URL,
        )

        username = st.text_input(
            "Username",
            value="standard_user",
        )

        password = st.text_input(
            "Password",
            type="password",
            value="secret_sauce",
        )

        st.divider()

        st.write(
            f"Model: `{OLLAMA_MODEL}`"
        )

        st.write(
            "Playwright MCP: `0.0.80`"
        )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    if "messages" not in st.session_state:

        st.session_state.messages = []


    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # ========================================================
    # USER PROMPT
    # ========================================================

    prompt = st.chat_input(
        "What should the browser do?"
    )

    if not prompt:
        return


    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )


    with st.chat_message(
        "user"
    ):

        st.write(
            prompt
        )


    # ========================================================
    # RUN AGENT
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        status_placeholder = st.empty()

        status_placeholder.write(
            "Running browser agent..."
        )

        try:

            result = asyncio.run(
                run_browser_agent(
                    instruction=prompt,
                    url=url,
                    username=username,
                    password=password,
                )
            )


            status_placeholder.write(
                "Browser task completed."
            )


            # =================================================
            # AGENT RESPONSE
            # =================================================

            agent_response = result.get(
                "agent_response"
            )

            if agent_response:

                st.write(
                    agent_response
                )

            else:

                st.write(
                    "Browser workflow completed."
                )


            st.divider()


            # =================================================
            # GENERATED FILES
            # =================================================

            st.subheader(
                "Generated files"
            )

            st.write(
                f"JSON: `{result['json_path']}`"
            )

            st.write(
                f"YAML: `{result['snapshot_path']}`"
            )

            st.write(
                f"Tool log: `{result['log_path']}`"
            )

            st.write(
                f"Final URL: `{result['url']}`"
            )


            # =================================================
            # SNAPSHOT
            # =================================================

            with st.expander(
                "Playwright browser_snapshot"
            ):

                st.code(
                    result["snapshot"],
                    language="yaml",
                )


            # =================================================
            # PAGE OBJECT JSON
            # =================================================

            with st.expander(
                "Page object JSON"
            ):

                st.json(
                    result["dom"]
                )


            # =================================================
            # TOOL ACTIVITY
            # =================================================

            with st.expander(
                "Playwright tool activity"
            ):

                render_activity(
                    result["activity"]
                )


            # =================================================
            # SAVE CHAT RESPONSE
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        agent_response
                        or
                        "Browser workflow completed."
                    ),
                }
            )


        except Exception as exc:

            status_placeholder.write(
                "Browser task failed."
            )

            st.error(
                str(exc)
            )

            st.exception(
                exc
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Browser task failed: {exc}"
                    ),
                }
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()