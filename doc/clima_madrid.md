```mermaid
sequenceDiagram
    participant U as Usuario
    participant F as Frontend
    participant M as MAF Server
    participant O as Azure OpenAI
    participant MC as MCP Client
    participant MS as MCP Server
    participant W as Weather API
    
    U->>F: "¿Qué clima hace en Madrid?"
    F->>M: POST /chatkit (user_message)
    
    M->>O: Send query to GPT-4o
    O->>O: Detecta necesidad de herramienta
    
    O-->>M: Response with tool call
    Note over O,M: {tool_calls: [{name: "get_current_weather", args: {location: "Madrid"}}]}
    
    M->>MC: Call MCP tool
    MC->>MS: JSON-RPC: tools/call
    MS->>W: HTTP GET weather API
    W-->>MS: Weather data
    MS-->>MC: Tool result
    
    MC-->>M: Formatted weather data
    
    M->>O: Send tool result to GPT-4o
    O->>O: Generate human-readable response
    
    O-->>M: Final response
    M-->>F: SSE: thread.item.done
    F-->>U: Muestra widget de clima```