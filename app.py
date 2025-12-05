# Copyright (c) Microsoft. All rights reserved.

"""
ChatKit Integration Sample with Weather Agent and Image Analysis

This sample demonstrates how to integrate Microsoft Agent Framework with OpenAI ChatKit
using a weather tool with widget visualization, image analysis, and Azure OpenAI. It shows
a complete ChatKit server implementation using Agent Framework agents with proper FastAPI
setup, interactive weather widgets, and vision capabilities for analyzing uploaded images.
"""

import logging
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from random import randint
from typing import Annotated, Any

import uvicorn

# Agent Framework imports
from agent_framework import AgentRunResponseUpdate, ChatAgent, ChatMessage, FunctionResultContent, Role
from agent_framework.azure import AzureOpenAIChatClient

# Agent Framework ChatKit integration
from agent_framework_chatkit import ThreadItemConverter, stream_agent_response

# Local imports
from attachment_store import FileBasedAttachmentStore
from azure.identity import AzureCliCredential

# ChatKit imports
from chatkit.actions import Action
from chatkit.server import ChatKitServer
from chatkit.store import StoreItemType, default_generate_id
from chatkit.types import (
    ThreadItem,
    ThreadItemDoneEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
    WidgetItem,
)
from chatkit.widgets import WidgetRoot
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import Field
from store import SQLiteStore
from weather_widget import (
    WeatherData,
    city_selector_copy_text,
    render_city_selector_widget,
    render_weather_widget,
    weather_widget_copy_text,
)

import os
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential

# Cargar tus variables del .env
load_dotenv()

# Obtener TUS variables
AZURE_ENDPOINT = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID") 
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")

# Verificar que todas las variables existan
if not all([AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET]):
    raise ValueError("Faltan variables de Azure en el archivo .env")

# Crear credencial con TUS variables
azure_credential = ClientSecretCredential(
    tenant_id=AZURE_TENANT_ID,
    client_id=AZURE_CLIENT_ID,
    client_secret=AZURE_CLIENT_SECRET
)

# ============================================================================
# Configuration Constants
# ============================================================================

# Server configuration
SERVER_HOST = "0.0.0.0"  # Bind to localhost only for security (local dev)
SERVER_PORT = 8001
SERVER_BASE_URL = f"http://localhost:{SERVER_PORT}"

# Database configuration
DATABASE_PATH = "chatkit_demo.db"

# File storage configuration
UPLOADS_DIRECTORY = "./uploads"

# User context
DEFAULT_USER_ID = "demo_user"

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)


class WeatherResponse(str):
    """A string response that also carries WeatherData for widget creation."""

    def __new__(cls, text: str, weather_data: WeatherData):
        instance = super().__new__(cls, text)
        instance.weather_data = weather_data  # type: ignore
        return instance


async def stream_widget(
    thread_id: str,
    widget: WidgetRoot,
    copy_text: str | None = None,
    generate_id: Callable[[StoreItemType], str] = default_generate_id,
) -> AsyncIterator[ThreadStreamEvent]:
    """Stream a ChatKit widget as a ThreadStreamEvent.

    This helper function creates a ChatKit widget item and yields it as a
    ThreadItemDoneEvent that can be consumed by the ChatKit UI.

    Args:
        thread_id: The ChatKit thread ID for the conversation.
        widget: The ChatKit widget to display.
        copy_text: Optional text representation of the widget for copy/paste.
        generate_id: Optional function to generate IDs for ChatKit items.

    Yields:
        ThreadStreamEvent: ChatKit event containing the widget.
    """
    item_id = generate_id("message")

    widget_item = WidgetItem(
        id=item_id,
        thread_id=thread_id,
        created_at=datetime.now(),
        widget=widget,
        copy_text=copy_text,
    )

    yield ThreadItemDoneEvent(type="thread.item.done", item=widget_item)


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location.

    Returns a string description with embedded WeatherData for widget creation.
    """
    logger.info(f"Fetching weather for location: {location}")

    conditions = ["sunny", "cloudy", "rainy", "stormy", "snowy", "foggy"]
    temperature = randint(-5, 35)
    condition = conditions[randint(0, len(conditions) - 1)]

    # Add some realistic details
    humidity = randint(30, 90)
    wind_speed = randint(5, 25)

    weather_data = WeatherData(
        location=location,
        condition=condition,
        temperature=temperature,
        humidity=humidity,
        wind_speed=wind_speed,
    )

    logger.debug(f"Weather data generated: {condition}, {temperature}°C, {humidity}% humidity, {wind_speed} km/h wind")

    # Return a WeatherResponse that is both a string (for the LLM) and carries structured data
    text = (
        f"Weather in {location}:\n"
        f"• Condition: {condition.title()}\n"
        f"• Temperature: {temperature}°C\n"
        f"• Humidity: {humidity}%\n"
        f"• Wind: {wind_speed} km/h"
    )
    return WeatherResponse(text, weather_data)


def get_time() -> str:
    """Get the current UTC time."""
    current_time = datetime.now(timezone.utc)
    logger.info("Getting current UTC time")
    return f"Current UTC time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"


def show_city_selector() -> str:
    """Show an interactive city selector widget to the user.

    This function triggers the display of a widget that allows users
    to select from popular cities to get weather information.

    Returns a special marker string that will be detected to show the widget.
    """
    logger.info("Activating city selector widget")
    return "__SHOW_CITY_SELECTOR__"




class WeatherChatKitServer(ChatKitServer[dict[str, Any]]):
    """ChatKit server implementation using Agent Framework.

    This server integrates Agent Framework agents with ChatKit's server protocol,
    providing weather information with interactive widgets and time queries through Azure OpenAI.
    """

    def __init__(self, data_store: SQLiteStore, attachment_store: FileBasedAttachmentStore):
        super().__init__(data_store, attachment_store)

        logger.info("Initializing WeatherChatKitServer")

        # Create Agent Framework agent with Azure OpenAI
        # For authentication, run `az login` command in terminal
        try:
            self.weather_agent = ChatAgent(
                chat_client = AzureOpenAIChatClient(
                    credential=azure_credential,  # ← USANDO TU CREDENCIAL
                    endpoint=AZURE_ENDPOINT,  # ← TU ENDPOINT
                    deployment_name=AZURE_DEPLOYMENT,  # ← TU DEPLOYMENT
                    api_version="2024-10-21",
                ),
                instructions=(
                    "You are a helpful weather assistant with image analysis capabilities. "
                    "You can provide weather information for any location, tell the current time, "
                    "and analyze images that users upload. Be friendly and informative in your responses.\n\n"
                    "If a user asks to see a list of cities or wants to choose from available cities, "
                    "use the show_city_selector tool to display an interactive city selector.\n\n"
                    "When users upload images, you will automatically receive them and can analyze their content. "
                    "Describe what you see in detail and be helpful in answering questions about the images."
                ),
                tools=[get_weather, get_time, show_city_selector],
            )
            logger.info("Weather agent initialized successfully with Azure OpenAI")
        except Exception as e:
            logger.error(f"Failed to initialize weather agent: {e}")
            raise

        # Create ThreadItemConverter with attachment data fetcher
        self.converter = ThreadItemConverter(
            attachment_data_fetcher=self._fetch_attachment_data,
        )

        logger.info("WeatherChatKitServer initialized")

    async def _fetch_attachment_data(self, attachment_id: str) -> bytes:
        """Fetch attachment binary data for the converter.

        Args:
            attachment_id: The ID of the attachment to fetch.

        Returns:
            The binary data of the attachment.
        """
        return await attachment_store.read_attachment_bytes(attachment_id)

    async def _update_thread_title(
        self, thread: ThreadMetadata, thread_items: list[ThreadItem], context: dict[str, Any]
    ) -> None:
        """Update thread title using LLM to generate a concise summary.

        Args:
            thread: The thread metadata to update.
            thread_items: All items in the thread.
            context: The context dictionary.
        """
        logger.info(f"Attempting to update thread title for thread: {thread.id}")

        if not thread_items:
            logger.debug("No thread items available for title generation")
            return

        # Collect user messages to understand the conversation topic
        user_messages: list[str] = []
        for item in thread_items:
            if isinstance(item, UserMessageItem) and item.content:
                for content_part in item.content:
                    if hasattr(content_part, "text") and isinstance(content_part.text, str):
                        user_messages.append(content_part.text)
                        break

        if not user_messages:
            logger.debug("No user messages found for title generation")
            return

        logger.debug(f"Found {len(user_messages)} user message(s) for title generation")

        try:
            # Use the agent's chat client to generate a concise title
            # Combine first few messages to capture the conversation topic
            conversation_context = "\n".join(user_messages[:3])

            title_prompt = [
                ChatMessage(
                    role=Role.USER,
                    text=(
                        f"Generate a very short, concise title (max 40 characters) for a conversation "
                        f"that starts with:\n\n{conversation_context}\n\n"
                        "Respond with ONLY the title, nothing else."
                    ),
                )
            ]

            # Use the chat client directly for a quick, lightweight call
            response = await self.weather_agent.chat_client.get_response(
                messages=title_prompt,
                temperature=0.3,
                max_tokens=20,
            )

            if response.messages and response.messages[-1].text:
                title = response.messages[-1].text.strip().strip('"').strip("'")
                # Ensure it's not too long
                if len(title) > 50:
                    title = title[:47] + "..."

                thread.title = title
                await self.store.save_thread(thread, context)
                logger.info(f"Updated thread {thread.id} title to: {title}")

        except Exception as e:
            logger.warning(f"Failed to generate thread title, using fallback: {e}")
            # Fallback to simple truncation
            first_message: str = user_messages[0]
            title: str = first_message[:50].strip()
            if len(first_message) > 50:
                title += "..."
            thread.title = title
            await self.store.save_thread(thread, context)
            logger.info(f"Updated thread {thread.id} title to (fallback): {title}")

    async def respond(
        self,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: dict[str, Any],
    ) -> AsyncIterator[ThreadStreamEvent]:
        """Handle incoming user messages and generate responses.

        This method converts ChatKit messages to Agent Framework format using ThreadItemConverter,
        runs the agent, converts the response back to ChatKit events using stream_agent_response,
        and creates interactive weather widgets when weather data is queried.
        """
        from agent_framework import FunctionResultContent

        if input_user_message is None:
            logger.debug("Received None user message, skipping")
            return

        logger.info(f"Processing message for thread: {thread.id}")

        try:
            # Track weather data and city selector flag for this request
            weather_data: WeatherData | None = None
            show_city_selector = False

            # Load full thread history from the store
            thread_items_page = await self.store.load_thread_items(
                thread_id=thread.id,
                after=None,
                limit=1000,
                order="asc",
                context=context,
            )
            thread_items = thread_items_page.data

            # Convert ALL thread items to Agent Framework ChatMessages using ThreadItemConverter
            # This ensures the agent has the full conversation context
            agent_messages = await self.converter.to_agent_input(thread_items)

            if not agent_messages:
                logger.warning("No messages after conversion")
                return

            logger.info(f"Running agent with {len(agent_messages)} message(s)")

            # Run the Agent Framework agent with streaming
            agent_stream = self.weather_agent.run_stream(agent_messages)

            # Create an intercepting stream that extracts function results while passing through updates
            async def intercept_stream() -> AsyncIterator[AgentRunResponseUpdate]:
                nonlocal weather_data, show_city_selector
                async for update in agent_stream:
                    # Check for function results in the update
                    if update.contents:
                        for content in update.contents:
                            if isinstance(content, FunctionResultContent):
                                result = content.result

                                # Check if it's a WeatherResponse (string subclass with weather_data attribute)
                                if isinstance(result, str) and hasattr(result, "weather_data"):
                                    extracted_data = getattr(result, "weather_data", None)
                                    if isinstance(extracted_data, WeatherData):
                                        weather_data = extracted_data
                                        logger.info(f"Weather data extracted: {weather_data.location}")
                                # Check if it's the city selector marker
                                elif isinstance(result, str) and result == "__SHOW_CITY_SELECTOR__":
                                    show_city_selector = True
                                    logger.info("City selector flag detected")
                    yield update

            # Stream updates as ChatKit events with interception
            async for event in stream_agent_response(
                intercept_stream(),
                thread_id=thread.id,
            ):
                yield event

            # If weather data was collected during the tool call, create a widget
            if weather_data is not None and isinstance(weather_data, WeatherData):
                logger.info(f"Creating weather widget for location: {weather_data.location}")
                # Create weather widget
                widget = render_weather_widget(weather_data)
                copy_text = weather_widget_copy_text(weather_data)

                # Stream the widget
                async for widget_event in stream_widget(thread_id=thread.id, widget=widget, copy_text=copy_text):
                    yield widget_event
                logger.debug("Weather widget streamed successfully")

            # If city selector should be shown, create and stream that widget
            if show_city_selector:
                logger.info("Creating city selector widget")
                # Create city selector widget
                selector_widget = render_city_selector_widget()
                selector_copy_text = city_selector_copy_text()

                # Stream the widget
                async for widget_event in stream_widget(
                    thread_id=thread.id, widget=selector_widget, copy_text=selector_copy_text
                ):
                    yield widget_event
                logger.debug("City selector widget streamed successfully")

            # Update thread title based on first user message if not already set
            if not thread.title or thread.title == "New thread":
                await self._update_thread_title(thread, thread_items, context)

            logger.info(f"Completed processing message for thread: {thread.id}")

        except Exception as e:
            logger.error(f"Error processing message for thread {thread.id}: {e}", exc_info=True)

    async def action(
        self,
        thread: ThreadMetadata,
        action: Action[str, Any],
        sender: WidgetItem | None,
        context: dict[str, Any],
    ) -> AsyncIterator[ThreadStreamEvent]:
        """Handle widget actions from the frontend.

        This method processes actions triggered by interactive widgets,
        such as city selection from the city selector widget.
        """

        logger.info(f"Received action: {action.type} for thread: {thread.id}")

        if action.type == "city_selected":
            # Extract city information from the action payload
            city_label = action.payload.get("city_label", "Unknown")

            logger.info(f"City selected: {city_label}")
            logger.debug(f"Action payload: {action.payload}")

            # Track weather data for this request
            weather_data: WeatherData | None = None

            # Create an agent message asking about the weather
            agent_messages = [ChatMessage(role=Role.USER, text=f"What's the weather in {city_label}?")]

            logger.debug(f"Processing weather query: {agent_messages[0].text}")

            # Run the Agent Framework agent with streaming
            agent_stream = self.weather_agent.run_stream(agent_messages)

            # Create an intercepting stream that extracts function results while passing through updates
            async def intercept_stream() -> AsyncIterator[AgentRunResponseUpdate]:
                nonlocal weather_data
                async for update in agent_stream:
                    # Check for function results in the update
                    if update.contents:
                        for content in update.contents:
                            if isinstance(content, FunctionResultContent):
                                result = content.result

                                # Check if it's a WeatherResponse (string subclass with weather_data attribute)
                                if isinstance(result, str) and hasattr(result, "weather_data"):
                                    extracted_data = getattr(result, "weather_data", None)
                                    if isinstance(extracted_data, WeatherData):
                                        weather_data = extracted_data
                                        logger.info(f"Weather data extracted: {weather_data.location}")
                    yield update

            # Stream updates as ChatKit events with interception
            async for event in stream_agent_response(
                intercept_stream(),
                thread_id=thread.id,
            ):
                yield event

            # If weather data was collected during the tool call, create a widget
            if weather_data is not None and isinstance(weather_data, WeatherData):
                logger.info(f"Creating weather widget for: {weather_data.location}")
                # Create weather widget
                widget = render_weather_widget(weather_data)
                copy_text = weather_widget_copy_text(weather_data)

                # Stream the widget
                async for widget_event in stream_widget(thread_id=thread.id, widget=widget, copy_text=copy_text):
                    yield widget_event
                logger.debug("Weather widget created successfully from action")
            else:
                logger.warning("No weather data available to create widget after action")


# FastAPI application setup
app = FastAPI(
    title="ChatKit Weather & Vision Agent",
    description="Weather and image analysis assistant powered by Agent Framework and Azure OpenAI",
    version="1.0.0",
)

# Añade ESTO justo después de crear la app
@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    logger.info(f"🔍 [{datetime.now().strftime('%H:%M:%S')}] {request.method} {request.url.path}")
    logger.info(f"   Origin: {request.headers.get('origin', 'None')}")
    logger.info(f"   Content-Type: {request.headers.get('content-type', 'None')}")
    
    # Si es OPTIONS, responder inmediatamente
    if request.method == "OPTIONS":
        logger.info("   Handling OPTIONS pre-flight")
        response = Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
            }
        )
        return response
    
    try:
        response = await call_next(request)
        
        # Añadir headers CORS a todas las respuestas
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("origin", "*")
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response
    except Exception as e:
        logger.error(f"❌ Middleware error: {e}")
        raise

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1",
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:8001"
    ],  # Lista explícita de orígenes permitidos
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],  # Explícito
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache de pre-flight por 10 minutos
)

# Initialize data store and ChatKit server
logger.info("Initializing application components")
data_store = SQLiteStore(db_path=DATABASE_PATH)
attachment_store = FileBasedAttachmentStore(
    uploads_dir=UPLOADS_DIRECTORY,
    base_url=SERVER_BASE_URL,
    data_store=data_store,
)
chatkit_server = WeatherChatKitServer(data_store, attachment_store)
logger.info("Application initialization complete")


@app.post("/chatkit")
async def chatkit_endpoint(request: Request):
    """Main ChatKit endpoint that handles all ChatKit requests."""
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"📥 Received {request.method} request from {client_host}")
    
    # Get headers for debugging
    headers = dict(request.headers)
    logger.info(f"Headers: { {k: v for k, v in headers.items() if k.lower() != 'authorization'} }")
    
    # Obtener el cuerpo de la solicitud
    request_body = await request.body()
    
    # ⬇⬇⬇ DEBUG COMPLETO ⬇⬇⬇
    logger.info("=" * 80)
    logger.info("RAW REQUEST INSPECTION:")
    logger.info(f"Content-Length: {len(request_body)} bytes")
    
    if request_body:
        # Try to decode as UTF-8
        try:
            body_str = request_body.decode('utf-8')
            logger.info(f"Body as string: {body_str[:500]}...")
            
            # Try to parse as JSON
            try:
                import json
                request_json = json.loads(body_str)
                logger.info("✅ Valid JSON detected")
                logger.info(f"Request type: {request_json.get('type', 'MISSING_TYPE')}")
                
                # Log specific fields based on type
                req_type = request_json.get('type')
                if req_type == 'threads.create':
                    logger.info("📝 Threads.create request details:")
                    logger.info(f"  Title: {request_json.get('params', {}).get('title', 'NO_TITLE')}")
                    logger.info(f"  Metadata: {request_json.get('params', {}).get('metadata', {})}")
                elif req_type == 'threads.add_user_message':
                    logger.info("💬 Threads.add_user_message request details:")
                    logger.info(f"  Thread ID: {request_json.get('params', {}).get('thread_id', 'NO_THREAD_ID')}")
                    content = request_json.get('params', {}).get('content', [])
                    logger.info(f"  Content items: {len(content)}")
                    for i, item in enumerate(content):
                        logger.info(f"    Item {i}: type={item.get('type')}, text={item.get('text', 'NO_TEXT')[:50]}...")
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON: {e}")
                logger.info(f"Raw body: {body_str}")
                
        except UnicodeDecodeError:
            logger.warning("Body is not valid UTF-8, showing hex dump:")
            logger.info(request_body[:200].hex())
    else:
        logger.warning("Empty request body")
    
    logger.info("=" * 80)
    # ⬆⬆⬆ FIN DEL DEBUG CODE ⬆⬆⬆
    
    # Create context
    context = {"request": request}

    try:
        # Process the request using ChatKit server
        logger.info("🔄 Processing request through ChatKit server...")
        result = await chatkit_server.process(request_body, context)
        logger.info("✅ Request processed successfully")

        # Return appropriate response type
        if hasattr(result, "__aiter__"):  # StreamingResult
            logger.info("📤 Returning streaming response")
            
            async def logging_stream():
                chunk_count = 0
                async for chunk in result:
                    chunk_count += 1
                    if chunk_count <= 3:  # Log first 3 chunks for debugging
                        try:
                            if hasattr(chunk, 'decode'):
                                chunk_str = chunk.decode('utf-8', errors='ignore')
                                logger.debug(f"Stream chunk {chunk_count}: {chunk_str[:100]}...")
                        except:
                            pass
                    yield chunk
                
                logger.info(f"📊 Stream completed: {chunk_count} chunks sent")
            
            return StreamingResponse(
                logging_stream(), 
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:  # NonStreamingResult
            logger.info("📤 Returning non-streaming response")
            
            # Log the response
            if hasattr(result, 'json'):
                try:
                    import json as json_module
                    response_json = json_module.loads(result.json)
                    logger.info(f"Response: {json_module.dumps(response_json, indent=2)[:500]}...")
                except:
                    logger.info(f"Response (raw): {result.json[:500]}...")
            
            return Response(
                content=result.json, 
                media_type="application/json",
                headers={"Cache-Control": "no-cache"}
            )
            
    except Exception as e:
        logger.error(f"❌ ERROR processing ChatKit request:")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Error message: {str(e)}")
        logger.error(f"   Traceback:", exc_info=True)
        
        # Return user-friendly error
        import json as json_module
        error_response = {
            "error": True,
            "type": type(e).__name__,
            "message": str(e),
            "request_type": request_json.get('type', 'UNKNOWN') if 'request_json' in locals() else 'UNKNOWN',
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )

@app.post("/simple-test")
async def simple_test(request: Request):
    """Simple test endpoint to check if POST works."""
    logger.info("✅ Simple test endpoint called via POST")
    return {
        "success": True,
        "message": "POST is working!",
        "timestamp": datetime.now().isoformat(),
        "method": request.method,
        "headers": dict(request.headers)
    }

@app.get("/simple-test")
async def simple_test_get():
    """GET version for browser testing."""
    return {
        "success": True,
        "message": "GET is working! Use POST for the real test.",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/test-create-thread")
async def test_create_thread():
    """Test endpoint to create a thread directly."""
    import json
    from datetime import datetime
    
    test_data = {
        "type": "threads.create",
        "params": {
            "title": "Test Thread from API",
            "metadata": {"test": True, "timestamp": datetime.now().isoformat()}
        }
    }
    
    # Simulate processing
    try:
        # Create a mock thread
        from chatkit.types import ThreadMetadata
        
        thread_id = f"test-{datetime.now().timestamp()}"
        thread = ThreadMetadata(
            id=thread_id,
            created_at=datetime.now(),
            title=test_data["params"]["title"],
            metadata=test_data["params"]["metadata"]
        )
        
        return {
            "success": True,
            "message": "Thread created successfully (test endpoint)",
            "data": {
                "id": thread.id,
                "title": thread.title,
                "created_at": thread.created_at.isoformat(),
                "metadata": thread.metadata
            },
            "test_request": test_data
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_request": test_data
        }

@app.get("/health")
async def health_check():
    """Health check endpoint to verify server is running."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "server": "ChatKit Weather Agent",
        "version": "1.0.0"
    }

@app.get("/test-chatkit")
async def test_chatkit():
    """Test endpoint to verify ChatKit functionality."""
    from chatkit.types import ThreadMetadata
    from datetime import datetime
    
    # Create a test thread
    test_thread = ThreadMetadata(
        id="test-thread-" + datetime.now().isoformat(),
        created_at=datetime.now(),
        title="Test Thread",
        metadata={}
    )
    
    return {
        "message": "ChatKit server is running",
        "thread": {
            "id": test_thread.id,
            "title": test_thread.title,
            "created_at": test_thread.created_at.isoformat()
        },
        "endpoints": {
            "chatkit": "/chatkit (POST)",
            "health": "/health (GET)",
            "upload": "/upload/{attachment_id} (POST)",
            "preview": "/preview/{attachment_id} (GET)"
        }
    }


@app.post("/upload/{attachment_id}")
async def upload_file(attachment_id: str, file: UploadFile = File(...)):
    """Handle file upload for two-phase upload.

    The client POSTs the file bytes here after creating the attachment
    via the ChatKit attachments.create endpoint.
    """
    logger.info(f"Receiving file upload for attachment: {attachment_id}")

    try:
        # Read file contents
        contents = await file.read()

        # Save to disk
        file_path = attachment_store.get_file_path(attachment_id)
        file_path.write_bytes(contents)

        logger.info(f"Saved {len(contents)} bytes to {file_path}")

        # Load the attachment metadata from the data store
        attachment = await data_store.load_attachment(attachment_id, {"user_id": DEFAULT_USER_ID})

        # Clear the upload_url since upload is complete
        attachment.upload_url = None

        # Save the updated attachment back to the store
        await data_store.save_attachment(attachment, {"user_id": DEFAULT_USER_ID})

        # Return the attachment metadata as JSON
        return JSONResponse(content=attachment.model_dump(mode="json"))

    except Exception as e:
        logger.error(f"Error uploading file for attachment {attachment_id}: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Failed to upload file."})


@app.get("/preview/{attachment_id}")
async def preview_image(attachment_id: str):
    """Serve image preview/thumbnail.

    For simplicity, this serves the full image. In production, you should
    generate and cache thumbnails.
    """
    logger.debug(f"Serving preview for attachment: {attachment_id}")

    try:
        file_path = attachment_store.get_file_path(attachment_id)

        if not file_path.exists():
            return JSONResponse(status_code=404, content={"error": "File not found"})

        # Determine media type from file extension or attachment metadata
        # For simplicity, we'll try to load from the store
        try:
            attachment = await data_store.load_attachment(attachment_id, {"user_id": DEFAULT_USER_ID})
            media_type = attachment.mime_type
        except Exception:
            # Default to binary if we can't determine
            media_type = "application/octet-stream"

        return FileResponse(file_path, media_type=media_type)

    except Exception as e:
        logger.error(f"Error serving preview for attachment {attachment_id}: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Error serving preview for attachment."})


if __name__ == "__main__":
    # Run the server
    logger.info(f"Starting ChatKit Weather Agent server on {SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
