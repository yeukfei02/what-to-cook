"""
Tool for managing memories in Bedrock AgentCore Memory Service.

This module provides Bedrock AgentCore Memory capabilities with memory record
creation and retrieval.

Key Features:
------------
1. Event Management:
   • create_event: Store events in memory sessions

2. Memory Record Operations:
   • retrieve_memory_records: Semantic search for extracted memories
   • list_memory_records: List all memory records
   • get_memory_record: Get specific memory record
   • delete_memory_record: Delete memory records

Usage Examples:
--------------
```python
from strands import Agent
from strands_tools.agent_core_memory import AgentCoreMemoryToolProvider

# Initialize with required parameters
provider = AgentCoreMemoryToolProvider(
    memory_id="memory-123abc",  # Required
    actor_id="user-456",        # Required
    session_id="session-789",   # Required
    namespace="default",        # Required
)

agent = Agent(tools=provider.tools)

# Create a memory using the default IDs from initialization
agent.tool.agent_core_memory(
    action="record",
    content="Hello, Remeber that my current hobby is knitting?"
)

# Search memory records using the default namespace from initialization
agent.tool.agent_core_memory(
    action="retrieve",
    query="user preferences"
)
```
"""

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

import boto3
from boto3.session import Session as Boto3Session
from botocore.config import Config as BotocoreConfig
from strands import tool
from strands.types.tools import AgentTool


# Define memory actions as an Enum
class MemoryAction(str, Enum):
    """Enum for memory actions."""

    RECORD = "record"
    RETRIEVE = "retrieve"
    LIST = "list"
    GET = "get"
    DELETE = "delete"


# Define required parameters for each action
REQUIRED_PARAMS = {
    # Action names
    MemoryAction.RECORD: ["memory_id", "actor_id", "session_id", "content"],
    MemoryAction.RETRIEVE: ["memory_id", "namespace", "query"],
    MemoryAction.LIST: ["memory_id"],
    MemoryAction.GET: ["memory_id", "memory_record_id"],
    MemoryAction.DELETE: ["memory_id", "memory_record_id"],
}

# Set up logging
logger = logging.getLogger(__name__)

# Default region if not specified
DEFAULT_REGION = "us-west-2"


class AgentCoreMemoryToolProvider:
    """Provider for AgentCore Memory Service tools."""

    def __init__(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        namespace: str,
        region: Optional[str] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
        boto_session: Optional[Boto3Session] = None,
    ):
        """
        Initialize the AgentCore Memory tool provider.

        Args:
            memory_id: Memory ID to use for operations (required)
            actor_id: Actor ID to use for operations (required)
            session_id: Session ID to use for operations (required)
            namespace: Namespace for memory record operations (required)
            region: AWS region for the service
            boto_client_config: Optional boto client configuration
            boto_session: Optional boto3 Session for custom credentials and configuration.
                          If provided, this session will be used to create the AWS clients
                          instead of the default boto3 client.

        Raises:
            ValueError: If any of the required parameters are missing or empty
        """
        # Validate required parameters
        if not memory_id:
            raise ValueError("memory_id is required")
        if not actor_id:
            raise ValueError("actor_id is required")
        if not session_id:
            raise ValueError("session_id is required")
        if not namespace:
            raise ValueError("namespace is required")

        self.memory_id = memory_id
        self.actor_id = actor_id
        self.session_id = session_id
        self.namespace = namespace
        self.boto_session = boto_session

        # Set up client configuration with user agent
        if boto_client_config:
            existing_user_agent = getattr(boto_client_config, "user_agent_extra", None)
            # Append 'strands-agents-memory' to existing user_agent_extra or set it if not present
            if existing_user_agent:
                new_user_agent = f"{existing_user_agent} strands-agents-memory"
            else:
                new_user_agent = "strands-agents-memory"
            self.client_config = boto_client_config.merge(BotocoreConfig(user_agent_extra=new_user_agent))
        else:
            self.client_config = BotocoreConfig(user_agent_extra="strands-agents-memory")

        # Initialize the client

        # Resolve region from parameters, environment, or default
        self.region = region or DEFAULT_REGION

        # Initialize client with the appropriate region
        # Use boto3 Session if provided, otherwise use boto3 directly
        if self.boto_session:
            self.bedrock_agent_core_client = self.boto_session.client(
                "bedrock-agentcore",
                region_name=self.region,
                config=self.client_config,
            )
        else:
            self.bedrock_agent_core_client = boto3.client(
                "bedrock-agentcore",
                region_name=self.region,
                config=self.client_config,
            )

    @property
    def tools(self) -> list[AgentTool]:
        """Extract all @tool decorated methods from this instance."""
        tools = []

        for attr_name in dir(self):
            if attr_name == "tools":
                continue
            attr = getattr(self, attr_name)
            # Also check the original way for regular AgentTool instances
            if isinstance(attr, AgentTool):
                tools.append(attr)

        return tools

    @tool
    def agent_core_memory(
        self,
        action: str,
        content: Optional[str] = None,
        query: Optional[str] = None,
        memory_record_id: Optional[str] = None,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
    ) -> Dict:
        """
        Work with agent memories - create, search, retrieve, list, and manage memory records.

        This tool helps agents store and access memories, allowing them to remember important
        information across conversations and interactions.

        Key Capabilities:
        - Store new memories (text conversations or structured data)
        - Search for memories using semantic search
        - Browse and list all stored memories
        - Retrieve specific memories by ID
        - Delete unwanted memories

        Supported Actions:
        -----------------
        Memory Management:
        - record: Store a new memory (conversation or data)
          Use this when you need to save information for later recall.

        - retrieve: Find relevant memories using semantic search
          Use this when searching for specific information in memories.
          This is the best action for queries like "find memories about X" or "search for memories related to Y".

        - list: Browse all stored memories
          Use this to see all available memories without filtering.
          This is useful for getting an overview of what's been stored.

        - get: Fetch a specific memory by ID
          Use this when you already know the exact memory ID.

        - delete: Remove a specific memory
          Use this to delete memories that are no longer needed.

        Args:
            action: The memory operation to perform (one of: "record", "retrieve", "list", "get", "delete")
            content: For record action: Simple text string to store as a memory
                     Example: "User prefers vegetarian pizza with extra cheese"
            query: Search terms for finding relevant memories (required for retrieve action)
            memory_record_id: ID of a specific memory (required for get and delete actions)
            max_results: Maximum number of results to return (optional)
            next_token: Pagination token (optional)

        Returns:
            Dict: Response containing the requested memory information or operation status
        """
        try:
            # Use values from initialization
            memory_id = self.memory_id
            actor_id = self.actor_id
            session_id = self.session_id
            namespace = self.namespace

            # Use provided values or defaults for other parameters
            memory_record_id = memory_record_id
            max_results = max_results

            # Try to convert string action to Enum
            try:
                action_enum = MemoryAction(action)
            except ValueError:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"Action '{action}' is not supported. "
                            f"Supported actions: {', '.join([a.value for a in MemoryAction])}"
                        }
                    ],
                }

            # Validate required parameters

            # Create a dictionary mapping parameter names to their values
            param_values = {
                "memory_id": self.memory_id,
                "actor_id": self.actor_id,
                "session_id": self.session_id,
                "namespace": self.namespace,
                "content": content,
                "query": query,
                "memory_record_id": memory_record_id,
                "max_results": max_results,
                "next_token": next_token,
            }

            # Check which required parameters are missing
            missing_params = [param for param in REQUIRED_PARAMS[action_enum] if not param_values.get(param)]

            if missing_params:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"The following parameters are required for {action_enum.value} action: "
                                f"{', '.join(missing_params)}"
                            )
                        }
                    ],
                }

            # Execute the appropriate action
            try:
                # Handle action names by mapping to API methods
                if action_enum == MemoryAction.RECORD:
                    response = self.create_event(
                        memory_id=memory_id,
                        actor_id=actor_id,
                        session_id=session_id,
                        content=content,
                    )
                    # Extract only the relevant "event" field from the response
                    event_data = response.get("event", {}) if isinstance(response, dict) else {}
                    return {
                        "status": "success",
                        "content": [{"text": f"Memory created successfully: {json.dumps(event_data, default=str)}"}],
                    }
                elif action_enum == MemoryAction.RETRIEVE:
                    response = self.retrieve_memory_records(
                        memory_id=memory_id,
                        namespace=namespace,
                        search_query=query,
                        max_results=max_results,
                        next_token=next_token,
                    )
                    # Extract only the relevant fields from the response
                    relevant_data = {}
                    if isinstance(response, dict):
                        if "memoryRecordSummaries" in response:
                            relevant_data["memoryRecordSummaries"] = response["memoryRecordSummaries"]
                        if "nextToken" in response:
                            relevant_data["nextToken"] = response["nextToken"]

                    return {
                        "status": "success",
                        "content": [
                            {"text": f"Memories retrieved successfully: {json.dumps(relevant_data, default=str)}"}
                        ],
                    }
                elif action_enum == MemoryAction.LIST:
                    response = self.list_memory_records(
                        memory_id=memory_id,
                        namespace=namespace,
                        max_results=max_results,
                        next_token=next_token,
                    )
                    # Extract only the relevant fields from the response
                    relevant_data = {}
                    if isinstance(response, dict):
                        if "memoryRecordSummaries" in response:
                            relevant_data["memoryRecordSummaries"] = response["memoryRecordSummaries"]
                        if "nextToken" in response:
                            relevant_data["nextToken"] = response["nextToken"]

                    return {
                        "status": "success",
                        "content": [
                            {"text": f"Memories listed successfully: {json.dumps(relevant_data, default=str)}"}
                        ],
                    }
                elif action_enum == MemoryAction.GET:
                    response = self.get_memory_record(
                        memory_id=memory_id,
                        memory_record_id=memory_record_id,
                    )
                    # Extract only the relevant "memoryRecord" field from the response
                    memory_record = response.get("memoryRecord", {}) if isinstance(response, dict) else {}
                    return {
                        "status": "success",
                        "content": [
                            {"text": f"Memory retrieved successfully: {json.dumps(memory_record, default=str)}"}
                        ],
                    }
                elif action_enum == MemoryAction.DELETE:
                    response = self.delete_memory_record(
                        memory_id=memory_id,
                        memory_record_id=memory_record_id,
                    )
                    # Extract only the relevant "memoryRecordId" field from the response
                    memory_record_id = response.get("memoryRecordId", "") if isinstance(response, dict) else ""

                    return {
                        "status": "success",
                        "content": [{"text": f"Memory deleted successfully: {memory_record_id}"}],
                    }
            except Exception as e:
                error_msg = f"API error: {str(e)}"
                logger.error(error_msg)
                return {"status": "error", "content": [{"text": error_msg}]}

        except Exception as e:
            logger.error(f"Unexpected error in agent_core_memory tool: {str(e)}")
            return {"status": "error", "content": [{"text": str(e)}]}

    def create_event(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        content: str,
        event_timestamp: Optional[datetime] = None,
    ) -> Dict:
        """
        Create an event in a memory session.

        Creates a new event record in the specified memory session. Events are immutable
        records that capture interactions or state changes in your application.

        Args:
            memory_id: ID of the memory store
            actor_id: ID of the actor (user, agent, etc.) creating the event
            session_id: ID of the session this event belongs to
            payload: Text content to store as a memory
            event_timestamp: Optional timestamp for the event (defaults to current time)

        Returns:
            Dict: Response containing the created event details

        Raises:
            ValueError: If required parameters are invalid
            RuntimeError: If the API call fails
        """

        # Set default timestamp if not provided
        if event_timestamp is None:
            event_timestamp = datetime.now(timezone.utc)

        # Format the payload for the API
        formatted_payload = [{"conversational": {"content": {"text": content}, "role": "ASSISTANT"}}]

        return self.bedrock_agent_core_client.create_event(
            memoryId=memory_id,
            actorId=actor_id,
            sessionId=session_id,
            eventTimestamp=event_timestamp,
            payload=formatted_payload,
        )

    def retrieve_memory_records(
        self,
        memory_id: str,
        namespace: str,
        search_query: str,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
    ) -> Dict:
        """
        Retrieve memory records using semantic search.

        Performs a semantic search across memory records in the specified namespace,
        returning records that semantically match the search query. Results are ranked
        by relevance to the query.

        Args:
            memory_id: ID of the memory store to search in
            namespace: Namespace to search within (e.g., "actor/user123/userId")
            search_query: Natural language query to search for
            max_results: Maximum number of results to return (default: service default)
            next_token: Pagination token for retrieving additional results

        Returns:
            Dict: Response containing matching memory records and optional next_token
        """
        # Prepare request parameters
        params = {"memoryId": memory_id, "namespace": namespace, "searchCriteria": {"searchQuery": search_query}}
        if max_results is not None:
            params["maxResults"] = max_results
        if next_token is not None:
            params["nextToken"] = next_token

        return self.bedrock_agent_core_client.retrieve_memory_records(**params)

    def get_memory_record(
        self,
        memory_id: str,
        memory_record_id: str,
    ) -> Dict:
        """Get a specific memory record."""
        return self.bedrock_agent_core_client.get_memory_record(
            memoryId=memory_id,
            memoryRecordId=memory_record_id,
        )

    def list_memory_records(
        self,
        memory_id: str,
        namespace: str,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
    ) -> Dict:
        """List memory records."""
        params = {"memoryId": memory_id}
        if namespace is not None:
            params["namespace"] = namespace
        if max_results is not None:
            params["maxResults"] = max_results
        if next_token is not None:
            params["nextToken"] = next_token
        return self.bedrock_agent_core_client.list_memory_records(**params)

    def delete_memory_record(
        self,
        memory_id: str,
        memory_record_id: str,
    ) -> Dict:
        """Delete a specific memory record."""
        return self.bedrock_agent_core_client.delete_memory_record(
            memoryId=memory_id,
            memoryRecordId=memory_record_id,
        )
