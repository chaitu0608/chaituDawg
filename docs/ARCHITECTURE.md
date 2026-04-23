# Runtime Architecture

## End-to-end sequence

```mermaid
flowchart TD
  userMsg[UserMessage]
  limits[InputLimits]
  intent[IntentClassifier]
  route[RouteStateResolver]
  rag[LocalRAG]
  lead[LeadFlow]
  tool[LeadCaptureTool]
  memory[SimpleMemStore]
  response[AgentResponse]

  userMsg --> limits
  limits --> intent
  intent --> route
  route -->|inquiry| rag
  route -->|lead_flow| lead
  route -->|greeting| response
  rag --> memory
  lead -->|eligible| tool
  lead --> memory
  tool --> memory
  memory --> response
```

## Lead flow notes

- Slot order: `name -> email -> platform`
- One missing field is requested per turn.
- Email/platform retries are counted; fallback allows user to skip capture for now.
- Tool call is gated on complete validated payload.
- Capture uses payload fingerprint for idempotency and a timeout/circuit-breaker wrapper.
