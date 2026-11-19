"""
Defines structured workflow events for progress tracking.
These events are emitted by agents and displayed by the terminal visualizer.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class WorkflowEvent:
    """
    Base class for all workflow events.
    
    Attributes:
        event_type: Type of event ("extraction", "inventory", "pricing", "fulfillment")
        stage: Progress stage ("started", "in_progress", "completed", "failed")
        timestamp: When the event occurred
        description: Human-readable description of what happened
        request_id: Which customer request this relates to
        item_id: Optional - which catalog item (if applicable)
        details: Optional - additional structured data for analysis
    """
    
    event_type: str  # "extraction", "inventory", "pricing", "fulfillment"
    stage: str       # "started", "in_progress", "completed", "failed"
    timestamp: datetime
    description: str
    request_id: str
    item_id: Optional[str] = None
    details: Optional[dict] = field(default_factory=dict)
    
    def __str__(self):
        """Display-friendly string representation"""
        return f"[{self.event_type.upper()}] {self.description}"
    
    def to_dict(self):
        """Convert to dictionary for logging/storage"""
        return {
            "event_type": self.event_type,
            "stage": self.stage,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "request_id": self.request_id,
            "item_id": self.item_id,
            "details": self.details or {}
        }
